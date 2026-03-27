"""Cross-platform credential store for keystore passphrase caching.

Detects the best available backend at runtime. No hard dependency
on any OS-specific service — every backend is probed and the first
working one is used.

Backend priority:
  macOS      → Keychain Services (via keyring library)
  Windows    → Credential Locker / DPAPI (via keyring library)
  Linux      → Secret Service D-Bus > kernel keyctl
  Fallback   → None

Security note:
  We intentionally DO NOT provide an automatic encrypted-file fallback.
  In Hermes' current same-user execution model, any fallback whose key is
  derivable from local machine/user state would be reachable by the agent
  itself via file reads and local code execution, collapsing the security
  boundary around sealed secrets. If no real OS/keyctl-backed credential
  store exists, users must either:

    - type the keystore passphrase at startup (recommended), or
    - set HERMES_KEYSTORE_PASSPHRASE env var for headless/Docker/systemd
      deployments (conscious security tradeoff for unattended operation)
"""

import hashlib
import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SERVICE_NAME = "hermes-keystore"
_ACCOUNT_NAME = "master-passphrase"


# =========================================================================
# Backend ABC
# =========================================================================

class _Backend:
    """Abstract credential store backend."""
    name: str = "Unknown"

    def store(self, passphrase: str) -> bool:
        raise NotImplementedError

    def retrieve(self) -> Optional[str]:
        raise NotImplementedError

    def delete(self) -> bool:
        raise NotImplementedError


# =========================================================================
# Backend: keyring (macOS Keychain, Windows Credential Locker, Secret Service)
# =========================================================================

class _KeyringBackend(_Backend):
    """Cross-platform backend via the ``keyring`` library.

    Covers macOS Keychain, Windows Credential Locker, and Linux
    Secret Service (GNOME Keyring / KDE Wallet) if available.
    """

    def __init__(self, kr_module):
        self._kr = kr_module
        backend_obj = kr_module.get_keyring()
        raw_name = type(backend_obj).__name__
        _friendly = {
            "Keyring": "macOS Keychain",
            "KeyringBackend": "macOS Keychain",
            "WinVaultKeyring": "Windows Credential Locker",
            "SecretServiceKeyring": "Secret Service (GNOME/KDE)",
        }
        self.name = _friendly.get(raw_name, raw_name)

    def store(self, passphrase: str) -> bool:
        try:
            self._kr.set_password(_SERVICE_NAME, _ACCOUNT_NAME, passphrase)
            return True
        except Exception as e:
            logger.warning("keyring store failed: %s", e)
            return False

    def retrieve(self) -> Optional[str]:
        try:
            return self._kr.get_password(_SERVICE_NAME, _ACCOUNT_NAME)
        except Exception:
            return None

    def delete(self) -> bool:
        try:
            self._kr.delete_password(_SERVICE_NAME, _ACCOUNT_NAME)
            return True
        except Exception:
            return False


# =========================================================================
# Backend: Linux kernel keyring (keyctl)
# =========================================================================

class _KeyctlBackend(_Backend):
    """Linux kernel keyring via the ``keyctl`` userspace tool.

    Uses the per-UID *user* keyring (``@u``) which persists as long as
    the UID has running processes.  On systemd systems this means the
    passphrase survives across gateway restarts.

    The persistent keyring (``@us``) would survive logout but has a
    configurable idle expiry (default 3 days).  We use ``@u`` because
    gateway/cron services are long-running.
    """
    name = "Linux Kernel Keyring"
    _KEY_DESC = "hermes:keystore:passphrase"

    def store(self, passphrase: str) -> bool:
        try:
            result = subprocess.run(
                ["keyctl", "add", "user", self._KEY_DESC, passphrase, "@u"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    def retrieve(self) -> Optional[str]:
        try:
            result = subprocess.run(
                ["keyctl", "search", "@u", "user", self._KEY_DESC],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return None
            key_id = result.stdout.strip()
            result = subprocess.run(
                ["keyctl", "pipe", key_id],
                capture_output=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout:
                return result.stdout.decode("utf-8")
            return None
        except (OSError, subprocess.TimeoutExpired):
            return None

    def delete(self) -> bool:
        try:
            result = subprocess.run(
                ["keyctl", "search", "@u", "user", self._KEY_DESC],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return False
            key_id = result.stdout.strip()
            subprocess.run(
                ["keyctl", "revoke", key_id],
                capture_output=True, timeout=5,
            )
            return True
        except (OSError, subprocess.TimeoutExpired):
            return False


# =========================================================================
# Backend: Encrypted file (universal fallback)
# =========================================================================

class _EncryptedFileBackend(_Backend):
    """Encrypted file fallback — works everywhere, requires pynacl.

    Derives an encryption key from machine-id + UID + static salt via
    SHA-256 (simplified HKDF).  Security assumption: same user on same
    machine is trusted (equivalent to DPAPI on Windows).
    """
    name = "Encrypted File"

    def _derive_key(self) -> bytes:
        machine_id = _get_machine_id()
        uid = str(os.getuid()) if hasattr(os, "getuid") else os.getlogin()
        ikm = f"{machine_id}:{uid}:hermes-keystore-credential-v1".encode()
        return hashlib.sha256(ikm).digest()

    def _path(self) -> Path:
        hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
        return hermes_home / "keystore" / ".credential"

    def store(self, passphrase: str) -> bool:
        try:
            import nacl.secret
            import nacl.utils
            key = self._derive_key()
            box = nacl.secret.SecretBox(key)
            encrypted = box.encrypt(passphrase.encode("utf-8"))
            path = self._path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes(encrypted))
            os.chmod(str(path), 0o600)
            return True
        except Exception as e:
            logger.warning("Encrypted file store failed: %s", e)
            return False

    def retrieve(self) -> Optional[str]:
        try:
            import nacl.secret
            key = self._derive_key()
            box = nacl.secret.SecretBox(key)
            encrypted = self._path().read_bytes()
            return box.decrypt(encrypted).decode("utf-8")
        except Exception:
            return None

    def delete(self) -> bool:
        try:
            self._path().unlink()
            return True
        except OSError:
            return False


# =========================================================================
# Machine ID helper
# =========================================================================

def _get_machine_id() -> str:
    """Get a stable machine identifier.  Best-effort, never raises."""
    # Linux
    for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            with open(path) as f:
                mid = f.read().strip()
                if mid:
                    return mid
        except OSError:
            continue

    # macOS — IOPlatformUUID
    if platform.system() == "Darwin":
        try:
            r = subprocess.run(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                capture_output=True, text=True, timeout=5,
            )
            for line in r.stdout.splitlines():
                if "IOPlatformUUID" in line:
                    return line.split('"')[-2]
        except (OSError, subprocess.TimeoutExpired, IndexError):
            pass

    # Windows — WMI CSProduct UUID
    if platform.system() == "Windows":
        try:
            r = subprocess.run(
                ["wmic", "csproduct", "get", "UUID"],
                capture_output=True, text=True, timeout=5,
            )
            lines = [l.strip() for l in r.stdout.splitlines()
                     if l.strip() and l.strip() != "UUID"]
            if lines:
                return lines[0]
        except (OSError, subprocess.TimeoutExpired):
            pass

    # Last resort: hostname (stable-ish)
    return platform.node()


# =========================================================================
# Backend detection
# =========================================================================

def _detect_backend() -> Optional[_Backend]:
    """Detect the best available credential store backend."""

    # 1. keyring library (macOS Keychain, Windows Credential Locker,
    #    or Linux Secret Service via D-Bus)
    try:
        import keyring
        from keyring.backends import fail as fail_backend

        backend_obj = keyring.get_keyring()
        if isinstance(backend_obj, fail_backend.Keyring):
            raise ValueError("only fail backend available")
        # Chainer with only fail backends
        if hasattr(backend_obj, "backends"):
            real = [b for b in backend_obj.backends
                    if not isinstance(b, fail_backend.Keyring)]
            if not real:
                raise ValueError("chainer has no real backends")
        return _KeyringBackend(keyring)
    except (ImportError, ValueError, Exception) as e:
        logger.debug("keyring unavailable: %s", e)

    # 2. Linux kernel keyctl
    if platform.system() == "Linux":
        try:
            result = subprocess.run(
                ["keyctl", "--version"],
                capture_output=True, timeout=5,
            )
            if result.returncode == 0:
                return _KeyctlBackend()
        except (OSError, subprocess.TimeoutExpired):
            pass

    # No insecure fallback. If no real backend is available, return None.
    return None


# Module-level cached backend.  ``False`` = not yet detected.
_cached_backend: Optional[_Backend] = None
_detection_done: bool = False


def _get_backend() -> Optional[_Backend]:
    global _cached_backend, _detection_done
    if not _detection_done:
        _cached_backend = _detect_backend()
        _detection_done = True
        if _cached_backend:
            logger.debug("Credential store backend: %s", _cached_backend.name)
        else:
            logger.debug("No credential store backend available")
    return _cached_backend


# =========================================================================
# Public API
# =========================================================================

def is_available() -> bool:
    """Return True if any credential store backend is available."""
    return _get_backend() is not None


def backend_name() -> Optional[str]:
    """Return human-readable name of the detected backend, or None."""
    b = _get_backend()
    return b.name if b else None


def store_passphrase(passphrase: str) -> bool:
    """Store the keystore passphrase.  Returns True on success."""
    b = _get_backend()
    if b is None:
        return False
    return b.store(passphrase)


def retrieve_passphrase() -> Optional[str]:
    """Retrieve the stored passphrase, or None if unavailable."""
    b = _get_backend()
    if b is None:
        return None
    return b.retrieve()


def delete_passphrase() -> bool:
    """Delete the stored passphrase.  Returns True on success."""
    b = _get_backend()
    if b is None:
        return False
    return b.delete()
