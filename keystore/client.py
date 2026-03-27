"""High-level keystore client for CLI and agent integration.

This is the main entry point for all keystore consumers.  It wraps
EncryptedStore with:

- Automatic path resolution (``~/.hermes/keystore/secrets.db``)
- Unlock flow (credential store → env var → interactive prompt)
- Injectable secret injection into ``os.environ``
- .env migration helper
- Singleton pattern (one client per process)

Usage in CLI startup::

    from keystore.client import get_keystore

    ks = get_keystore()
    ks.ensure_unlocked()         # prompts if needed
    ks.inject_env()              # populates os.environ with injectable secrets

Usage in gateway startup::

    ks = get_keystore()
    ks.ensure_unlocked(interactive=False)  # raises if can't auto-unlock
    ks.inject_env()
"""

import getpass
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from keystore.store import (
    EncryptedStore,
    KeystoreError,
    KeystoreLocked,
    PassphraseMismatch,
    SecretEntry,
)
from keystore import credential_store
from keystore.categories import SecretCategory, default_category

logger = logging.getLogger(__name__)


def _hermes_home() -> Path:
    return Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))


def _default_db_path() -> Path:
    return _hermes_home() / "keystore" / "secrets.db"


def _env_file_path() -> Path:
    return _hermes_home() / ".env"


def _owned_env_names() -> set[str]:
    raw = os.getenv("HERMES_KEYSTORE_OWNED_VARS", "")
    return {x for x in raw.split(",") if x}


def _set_owned_env_names(names: set[str]) -> None:
    os.environ["HERMES_KEYSTORE_OWNED_VARS"] = ",".join(sorted(names))


def _owned_env_values() -> dict[str, str]:
    raw = os.getenv("HERMES_KEYSTORE_OWNED_VALUES_JSON", "")
    if not raw:
        return {}
    try:
        import json as _json
        data = _json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _set_owned_env_values(values: dict[str, str]) -> None:
    import json as _json
    os.environ["HERMES_KEYSTORE_OWNED_VALUES_JSON"] = _json.dumps(values, sort_keys=True)


class KeystoreClient:
    """High-level keystore interface for CLI, gateway, and agent startup."""

    def __init__(self, db_path: Optional[str | Path] = None):
        path = Path(db_path) if db_path else _default_db_path()
        self._store = EncryptedStore(path)
        self._injected: Dict[str, bool] = {}

    @property
    def is_initialized(self) -> bool:
        return self._store.is_initialized

    @property
    def is_unlocked(self) -> bool:
        return self._store.is_unlocked

    def initialize(self, passphrase: str) -> None:
        """Initialize a new keystore with the given passphrase."""
        self._store.initialize(passphrase)

    def unlock(self, passphrase: str) -> None:
        """Unlock with a known passphrase."""
        self._store.unlock(passphrase)

    def lock(self) -> None:
        """Lock the keystore."""
        self._store.lock()

    def ensure_unlocked(self, interactive: bool = True) -> bool:
        """Ensure the keystore is unlocked, trying all available methods.

        Unlock priority:
        1. Already unlocked → no-op
        2. OS credential store (if ``hermes keystore remember`` was used)
        3. Interactive passphrase prompt (if ``interactive=True``)
        4. ``HERMES_KEYSTORE_PASSPHRASE`` env var (headless/Docker fallback only)

        Returns True if unlocked, False if not initialized (caller should
        set up the keystore), raises PassphraseMismatch on wrong passphrase.

        When ``interactive=False`` (gateway/headless), raises KeystoreLocked
        if no automatic unlock method succeeds.
        """
        if self._store.is_unlocked:
            return True

        if not self._store.is_initialized:
            return False

        # 1. Try credential store
        passphrase = credential_store.retrieve_passphrase()
        if passphrase:
            try:
                self._store.unlock(passphrase)
                logger.debug("Unlocked via credential store (%s)", credential_store.backend_name())
                return True
            except PassphraseMismatch:
                logger.warning(
                    "Stored passphrase is stale (credential store: %s). "
                    "Run 'hermes keystore remember' to update it.",
                    credential_store.backend_name(),
                )

        # 2. Interactive prompt (preferred over env var when TTY is available)
        if interactive:
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    passphrase = getpass.getpass("🔐 Keystore passphrase: ")
                    if not passphrase:
                        continue
                    self._store.unlock(passphrase)
                    return True
                except PassphraseMismatch:
                    remaining = max_attempts - attempt - 1
                    if remaining > 0:
                        print(f"  ✗ Incorrect passphrase ({remaining} attempts remaining)")
                    else:
                        print("  ✗ Incorrect passphrase")

            raise PassphraseMismatch("Too many incorrect passphrase attempts")

        # 3. Env var — last resort for headless/Docker/systemd deployments
        # where no TTY or credential store is available. The passphrase is
        # visible in the process environment, so this is a conscious security
        # tradeoff for unattended operation.
        env_passphrase = os.getenv("HERMES_KEYSTORE_PASSPHRASE")
        if env_passphrase:
            try:
                self._store.unlock(env_passphrase)
                logger.debug("Unlocked via HERMES_KEYSTORE_PASSPHRASE env var (headless fallback)")
                return True
            except PassphraseMismatch:
                logger.warning("HERMES_KEYSTORE_PASSPHRASE env var has wrong passphrase")

        raise KeystoreLocked(
            "Keystore is locked and no automatic unlock method succeeded. "
            "Run 'hermes keystore remember' to cache the passphrase, or "
            "set HERMES_KEYSTORE_PASSPHRASE env var for headless deployments."
        )

    def inject_env(self, force: bool = False, external_managed_names: Optional[set[str]] = None) -> Dict[str, bool]:
        """Inject all ``injectable`` secrets into ``os.environ``.

        Args:
            force: Refresh mode for long-lived processes. When ``False``
                (default), existing env vars are preserved so shell/Docker env
                wins over keystore values at startup. When ``True``, only env
                vars that were previously injected by this client instance are
                refreshed.
            external_managed_names: Optional set of env-var names that were
                supplied by non-keystore sources during the current refresh
                cycle and explicitly passed in by the caller. In the current
                gateway implementation this is used for `.env`-tracked names.
                This lets long-lived processes distinguish a stale injected
                value from an external replacement even when the replacement
                uses the same credential string.

        Returns:
            Dict of ``{secret_name: injected_or_overwritten}``.
        """
        secrets = self._store.get_injectable_secrets()
        previous = dict(self._injected)
        owned = _owned_env_names()
        owned_values = _owned_env_values()
        external_managed_names = set(external_managed_names or set())
        injected = {}
        current_names = set(secrets.keys())

        # Force-refresh also acts as revocation for previously keystore-owned
        # env vars that have been deleted from the keystore or are no longer
        # injectable. Only revoke names that are still keystore-owned AND not
        # externally managed in this refresh cycle.
        if force:
            removed = owned - current_names
            for name in removed:
                if name not in external_managed_names:
                    os.environ.pop(name, None)
                owned.discard(name)
                owned_values.pop(name, None)

        for name, value in secrets.items():
            should_write = False
            if name not in os.environ:
                should_write = True
            elif name in external_managed_names:
                # Current refresh explicitly sourced this name externally.
                should_write = False
            elif force and (previous.get(name) is True or name in owned):
                # Only refresh vars we previously injected ourselves.
                should_write = True

            if should_write:
                os.environ[name] = value
                injected[name] = True
                owned.add(name)
                owned_values[name] = value
            else:
                injected[name] = False
                if name in external_managed_names and not (previous.get(name) is True):
                    # External source owns it in this process.
                    owned.discard(name)
                    owned_values.pop(name, None)
        self._injected = injected
        _set_owned_env_names(owned)
        _set_owned_env_values(owned_values)
        count_written = sum(1 for v in injected.values() if v)
        count_skipped = sum(1 for v in injected.values() if not v)
        logger.info(
            "Keystore: %s %d secrets (%d skipped)",
            "refreshed" if force else "injected",
            count_written,
            count_skipped,
        )
        return injected

    # ------------------------------------------------------------------
    # Secret management
    # ------------------------------------------------------------------

    def set_secret(
        self,
        name: str,
        value: str,
        category: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> None:
        """Store a secret.  Category defaults based on the name."""
        cat = category or default_category(name).value
        self._store.set(name, value, category=cat, description=description, tags=tags)

    def get_secret(self, name: str, requester: str = "cli") -> Optional[str]:
        """Retrieve a secret."""
        return self._store.get(name, requester=requester)

    def delete_secret(self, name: str) -> bool:
        """Delete a secret."""
        return self._store.delete(name)

    def list_secrets(self) -> List[SecretEntry]:
        """List all secrets (metadata only)."""
        return self._store.list_secrets()

    def set_category(self, name: str, category: str) -> bool:
        """Change a secret's access category."""
        # Validate
        try:
            SecretCategory(category)
        except ValueError:
            raise KeystoreError(
                f"Invalid category '{category}'. "
                f"Must be one of: {', '.join(c.value for c in SecretCategory)}"
            )
        return self._store.set_category(name, category)

    def get_access_log(self, limit: int = 50) -> List[dict]:
        """Return recent access log entries."""
        return self._store.get_access_log(limit)

    def change_passphrase(self, old_passphrase: str, new_passphrase: str) -> None:
        """Change the master passphrase."""
        self._store.change_passphrase(old_passphrase, new_passphrase)

    def secret_count(self) -> int:
        """Return the number of stored secrets."""
        return self._store.secret_count()

    # ------------------------------------------------------------------
    # Credential store (passphrase caching)
    # ------------------------------------------------------------------

    def remember_passphrase(self, passphrase: str) -> Tuple[bool, str]:
        """Store the passphrase in the OS credential store.

        Returns (success, backend_name_or_error_message).
        """
        backend = credential_store.backend_name()
        if not credential_store.is_available():
            return False, (
                "No credential store backend available.\n\n"
                "Options:\n"
                "  • Set HERMES_KEYSTORE_PASSPHRASE env var for headless/Docker\n"
                "  • Install keyring: pip install keyring\n"
                "  • Install keyctl: apt install keyutils (Linux)\n"
                "  • Type your passphrase each time (most secure)"
            )
        # Verify the passphrase is correct first
        try:
            self._store.unlock(passphrase)
        except PassphraseMismatch:
            return False, "Incorrect passphrase"

        if credential_store.store_passphrase(passphrase):
            return True, backend
        return False, f"Failed to store passphrase in {backend}"

    def forget_passphrase(self) -> Tuple[bool, str]:
        """Remove the passphrase from the OS credential store."""
        backend = credential_store.backend_name()
        if credential_store.delete_passphrase():
            return True, backend or "credential store"
        return False, "No stored passphrase found"

    # ------------------------------------------------------------------
    # Migration from .env
    # ------------------------------------------------------------------

    def migrate_from_env(self, env_path: Optional[Path] = None) -> Dict[str, str]:
        """Import secrets from a .env file into the keystore.

        Returns a dict of {secret_name: category} for each migrated secret.
        Skips blank values and comments.  Does NOT delete the .env file
        (the caller should handle backup/stub creation).
        """
        path = env_path or _env_file_path()
        if not path.exists():
            return {}

        migrated = {}
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue

                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()

                # Strip surrounding quotes
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                    value = value[1:-1]

                if not value:
                    continue

                # Skip non-secret config values
                if not _looks_like_secret(key, value):
                    continue

                category = default_category(key).value
                self._store.set(
                    key, value,
                    category=category,
                    description=f"Migrated from .env",
                    tags=["migrated"],
                )
                migrated[key] = category

        logger.info("Migrated %d secrets from %s", len(migrated), path)
        return migrated


def _looks_like_secret(key: str, value: str) -> bool:
    """Heuristic: does this .env entry look like a secret?"""
    secret_indicators = (
        "KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD",
        "AUTH", "CREDENTIAL", "API_KEY",
    )
    key_upper = key.upper()
    for indicator in secret_indicators:
        if indicator in key_upper:
            return True
    # Long random-looking values are probably secrets
    if len(value) >= 20 and not value.startswith("/") and not value.startswith("http"):
        return True
    return False


# =========================================================================
# Singleton
# =========================================================================

_instance: Optional[KeystoreClient] = None


def get_keystore(db_path: Optional[str | Path] = None) -> KeystoreClient:
    """Get the global keystore client (singleton per process)."""
    global _instance
    if _instance is None:
        _instance = KeystoreClient(db_path)
    return _instance


def reset_keystore() -> None:
    """Reset the global singleton (for testing)."""
    global _instance
    if _instance is not None:
        try:
            _instance.lock()
        except Exception:
            pass
    _instance = None
