"""Direct in-process MLX chat and embedding adapters."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import warnings
from types import SimpleNamespace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        msvcrt = None

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_MLX_MODEL = os.getenv(
    "LOCAL_MLX_MODEL",
    "mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit",
)
DEFAULT_LOCAL_MLX_AUX_MODEL = os.getenv("LOCAL_MLX_AUX_MODEL", DEFAULT_LOCAL_MLX_MODEL)
DEFAULT_LOCAL_MLX_EMBED_MODEL = os.getenv(
    "LOCAL_MLX_EMBED_MODEL",
    "mlx-community/nomicai-modernbert-embed-base-4bit",
)
DEFAULT_LOCAL_MLX_PORT = int(os.getenv("LOCAL_MLX_PORT", "53147"))
DEFAULT_LOCAL_MLX_BASE_URL = f"http://127.0.0.1:{DEFAULT_LOCAL_MLX_PORT}"

# TurboQuant KV cache compression
try:
    from agent.kv_compression import TurboQuantConfig, TurboQuantKVCache
    TURBOQUANT_AVAILABLE = True
except ImportError:
    TURBOQUANT_AVAILABLE = False
    TurboQuantConfig = None  # type: ignore
    TurboQuantKVCache = None  # type: ignore

_turboquant_config: Optional["TurboQuantConfig"] = None


def get_turboquant_config() -> Optional["TurboQuantConfig"]:
    """Get or create TurboQuant config from environment."""
    global _turboquant_config
    if _turboquant_config is not None:
        return _turboquant_config
    if not TURBOQUANT_AVAILABLE:
        return None
    _turboquant_config = TurboQuantConfig.from_env()
    return _turboquant_config


def _hermes_home() -> Path:
    return Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))


def _shared_local_mlx_enabled() -> bool:
    return os.getenv("LOCAL_MLX_SHARED_ENDPOINT", "1").strip().lower() not in {"0", "false", "no", "off"}


def _local_mlx_server_root() -> Path:
    return _hermes_home() / "local_mlx"


def _local_mlx_server_lock_path() -> Path:
    return _local_mlx_server_root() / ".server.lock"


def _local_mlx_server_log_path() -> Path:
    return _hermes_home() / "logs" / "local_mlx_server.log"


def _local_mlx_server_err_path() -> Path:
    return _hermes_home() / "logs" / "local_mlx_server.error.log"


def local_mlx_configured() -> bool:
    provider = (os.getenv("HERMES_INFERENCE_PROVIDER") or "").strip().lower()
    return provider == "mlx" or bool(os.getenv("LOCAL_MLX_MODEL"))


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"text", "input_text"}:
                out.append(str(part.get("text", "")))
            elif isinstance(part, dict) and part.get("type") == "image_url":
                out.append("[image omitted]")
            else:
                out.append(str(part))
        return "\n".join(piece for piece in out if piece).strip()
    if content is None:
        return ""
    return str(content)


def _normalize_messages(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [
        {
            "role": msg.get("role", "user"),
            "content": _content_to_text(msg.get("content")),
        }
        for msg in messages
    ]


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    stripped = _strip_fences(text)
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _strip_think_artifacts(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned.strip()


def _tool_instruction(tools: List[Dict[str, Any]]) -> str:
    specs: List[Dict[str, Any]] = []
    for tool in tools:
        fn = tool.get("function", {}) if isinstance(tool, dict) else {}
        if fn.get("name"):
            specs.append(
                {
                    "name": fn.get("name"),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
            )
    if not specs:
        return ""
    return (
        "You may call tools. If a tool is required, respond with ONLY valid JSON in this format:\n"
        '{"tool_calls":[{"name":"tool_name","arguments":{"arg":"value"}}]}\n'
        "Do not use markdown fences. If no tool is needed, answer normally.\n\n"
        f"Available tools:\n{json.dumps(specs, ensure_ascii=False, indent=2)}"
    )


def _recover_terminal_tool_calls(text: str, tools: List[Dict[str, Any]]) -> Optional[List[SimpleNamespace]]:
    if not text or not tools:
        return None
    tool_names = {
        str((tool.get("function", {}) or {}).get("name", ""))
        for tool in tools
        if isinstance(tool, dict)
    }
    if "terminal" not in tool_names:
        return None
    blocks = re.findall(r"```(?:bash|sh|zsh)?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    commands = [block.strip() for block in blocks if block and block.strip()]
    if not commands:
        return None
    recovered: List[SimpleNamespace] = []
    for command in commands:
        recovered.append(
            SimpleNamespace(
                id=f"mlx_call_{uuid.uuid4().hex[:10]}",
                type="function",
                function=SimpleNamespace(
                    name="terminal",
                    arguments=json.dumps({"command": command}, ensure_ascii=False),
                ),
            )
        )
    return recovered or None


class _CrossProcessFileLock:
    def __init__(self, path: Path):
        self._path = path
        self._fd = None

    def __enter__(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = open(self._path, "w")
        if fcntl:
            fcntl.flock(self._fd, fcntl.LOCK_EX)
        elif "msvcrt" in globals() and msvcrt:
            msvcrt.locking(self._fd.fileno(), msvcrt.LK_LOCK, 1)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fd is None:
            return
        if fcntl:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        elif "msvcrt" in globals() and msvcrt:
            try:
                msvcrt.locking(self._fd.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        self._fd.close()
        self._fd = None


@lru_cache(maxsize=1)
def _mlx_http_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


def _mlx_http_request(method: str, path: str, *, payload: Optional[Dict[str, Any]] = None, timeout: float = 5.0) -> Any:
    response = _mlx_http_session().request(
        method.upper(),
        f"{DEFAULT_LOCAL_MLX_BASE_URL}{path}",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _shared_local_mlx_healthy() -> bool:
    try:
        data = _mlx_http_request("GET", "/health", timeout=0.5)
    except Exception:
        return False
    return isinstance(data, dict) and data.get("status") == "ok"


def _spawn_shared_local_mlx_server() -> None:
    server_root = _local_mlx_server_root()
    server_root.mkdir(parents=True, exist_ok=True)
    _local_mlx_server_log_path().parent.mkdir(parents=True, exist_ok=True)

    command = [sys.executable, "-m", "agent.local_mlx_server"]
    env = os.environ.copy()
    env.setdefault("LOCAL_MLX_PORT", str(DEFAULT_LOCAL_MLX_PORT))
    env.setdefault("PYTHONUNBUFFERED", "1")

    with open(_local_mlx_server_log_path(), "a") as stdout, open(_local_mlx_server_err_path(), "a") as stderr:
        subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            close_fds=True,
        )


def _ensure_shared_local_mlx_server() -> bool:
    if not _shared_local_mlx_enabled():
        return False
    if _shared_local_mlx_healthy():
        return True

    with _CrossProcessFileLock(_local_mlx_server_lock_path()):
        if _shared_local_mlx_healthy():
            return True
        _spawn_shared_local_mlx_server()
        deadline = time.monotonic() + float(os.getenv("LOCAL_MLX_SERVER_BOOT_TIMEOUT_SECONDS", "15"))
        while time.monotonic() < deadline:
            if _shared_local_mlx_healthy():
                return True
            time.sleep(0.2)
    logger.warning("Shared local MLX endpoint did not become healthy; falling back to in-process inference")
    return False


def _deserialize_tool_calls(tool_calls: Any) -> Optional[List[SimpleNamespace]]:
    if not isinstance(tool_calls, list):
        return None
    out: List[SimpleNamespace] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        out.append(
            SimpleNamespace(
                id=str(tool_call.get("id", f"mlx_call_{uuid.uuid4().hex[:10]}")),
                type=str(tool_call.get("type", "function")),
                function=SimpleNamespace(
                    name=str(function.get("name", "")),
                    arguments=str(function.get("arguments", "{}")),
                ),
            )
        )
    return out or None


def _deserialize_chat_response(data: Dict[str, Any]) -> Any:
    choices: List[SimpleNamespace] = []
    for raw_choice in data.get("choices", []) if isinstance(data, dict) else []:
        raw_message = raw_choice.get("message", {}) if isinstance(raw_choice, dict) else {}
        message = SimpleNamespace(
            role=str(raw_message.get("role", "assistant")),
            content=raw_message.get("content"),
            tool_calls=_deserialize_tool_calls(raw_message.get("tool_calls")),
        )
        choices.append(
            SimpleNamespace(
                index=int(raw_choice.get("index", 0)),
                message=message,
                finish_reason=str(raw_choice.get("finish_reason", "stop")),
            )
        )
    usage_data = data.get("usage", {}) if isinstance(data, dict) else {}
    usage = SimpleNamespace(
        prompt_tokens=int(usage_data.get("prompt_tokens", 0)),
        completion_tokens=int(usage_data.get("completion_tokens", 0)),
        total_tokens=int(usage_data.get("total_tokens", 0)),
    )
    return SimpleNamespace(
        choices=choices,
        model=str(data.get("model", DEFAULT_LOCAL_MLX_MODEL)) if isinstance(data, dict) else DEFAULT_LOCAL_MLX_MODEL,
        usage=usage,
    )


def _recover_xml_tool_calls(text: str, tools: List[Dict[str, Any]]) -> Optional[List[SimpleNamespace]]:
    if not text or not tools:
        return None

    tool_names = {
        str((tool.get("function", {}) or {}).get("name", "")).strip()
        for tool in tools
        if isinstance(tool, dict)
    }
    tool_names.discard("")
    if not tool_names:
        return None

    recovered: List[SimpleNamespace] = []

    # First recover the documented XML wrapper format:
    # <tool_call>{"name":"read_file","arguments":{"path":"/tmp/x"}}</tool_call>
    for block in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.DOTALL | re.IGNORECASE):
        parsed = _extract_json_object(block)
        if not parsed:
            continue
        name = str(parsed.get("name", "")).strip()
        if not name or name not in tool_names:
            continue
        args = parsed.get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        recovered.append(
            SimpleNamespace(
                id=f"mlx_call_{uuid.uuid4().hex[:10]}",
                type="function",
                function=SimpleNamespace(
                    name=name,
                    arguments=json.dumps(args, ensure_ascii=False),
                ),
            )
        )

    if recovered:
        return recovered

    # Also recover direct XML tool tags emitted by some local models:
    # <read_file><path>/tmp/file</path></read_file>
    pattern = re.compile(r"<([A-Za-z_][\w\-]*)>\s*(.*?)\s*</\1>", re.DOTALL)
    for match in pattern.finditer(text):
        name = match.group(1).strip()
        if name not in tool_names:
            continue
        body = match.group(2)
        args: Dict[str, Any] = {}
        for arg_match in re.finditer(r"<([A-Za-z_][\w\-]*)>\s*(.*?)\s*</\1>", body, flags=re.DOTALL):
            arg_name = arg_match.group(1).strip()
            arg_value = arg_match.group(2).strip()
            if not arg_name:
                continue
            # Best-effort scalar coercion for common schema values.
            if re.fullmatch(r"-?\d+", arg_value):
                coerced: Any = int(arg_value)
            elif arg_value.lower() in {"true", "false"}:
                coerced = arg_value.lower() == "true"
            else:
                coerced = arg_value
            args[arg_name] = coerced

        recovered.append(
            SimpleNamespace(
                id=f"mlx_call_{uuid.uuid4().hex[:10]}",
                type="function",
                function=SimpleNamespace(
                    name=name,
                    arguments=json.dumps(args, ensure_ascii=False),
                ),
            )
        )

    if recovered:
        return recovered

    # Tolerate malformed direct XML where the outer closing tag is wrong or
    # omitted, as long as the tool name is clear and the inner arg tags are
    # well-formed:
    # <web_search><query>foo</query></query>
    opener = re.search(r"<([A-Za-z_][\w\-]*)>\s*", text)
    if opener:
        name = opener.group(1).strip()
        if name in tool_names and name != "tool_call":
            body = text[opener.end():]
            args: Dict[str, Any] = {}
            for arg_match in re.finditer(r"<([A-Za-z_][\w\-]*)>\s*(.*?)\s*</\1>", body, flags=re.DOTALL):
                arg_name = arg_match.group(1).strip()
                arg_value = arg_match.group(2).strip()
                if not arg_name:
                    continue
                if re.fullmatch(r"-?\d+", arg_value):
                    coerced: Any = int(arg_value)
                elif arg_value.lower() in {"true", "false"}:
                    coerced = arg_value.lower() == "true"
                else:
                    coerced = arg_value
                args[arg_name] = coerced

            if args:
                recovered.append(
                    SimpleNamespace(
                        id=f"mlx_call_{uuid.uuid4().hex[:10]}",
                        type="function",
                        function=SimpleNamespace(
                            name=name,
                            arguments=json.dumps(args, ensure_ascii=False),
                        ),
                    )
                )

    return recovered or None


class LocalMLXService:
    _instance: Optional["LocalMLXService"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._chat_models: Dict[str, Any] = {}
        self._embed_models: Dict[str, Any] = {}
        self._chat_lock = threading.Lock()
        self._embed_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "LocalMLXService":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _load_chat(self, model_name: str):
        with self._chat_lock:
            cached = self._chat_models.get(model_name)
            if cached is not None:
                return cached
            try:
                from mlx_lm import load
            except Exception as exc:
                raise RuntimeError("mlx_lm is not installed for direct MLX chat.") from exc
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"The tokenizer you are loading from '.*' with an incorrect regex pattern:.*",
                )
                model, tokenizer = load(model_name)
            cached = (model, tokenizer)
            self._chat_models[model_name] = cached
            logger.info("Loaded local MLX chat model: %s", model_name)
            return cached

    def _load_embedder(self, model_name: str):
        with self._embed_lock:
            cached = self._embed_models.get(model_name)
            if cached is not None:
                return cached
            try:
                from mlx_embeddings import load
            except Exception as exc:
                raise RuntimeError("mlx_embeddings is not installed for direct MLX embeddings.") from exc
            cached = load(model_name)
            self._embed_models[model_name] = cached
            logger.info("Loaded local MLX embedding model: %s", model_name)
            return cached

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        *,
        model_name: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Any:
        model_name = model_name or DEFAULT_LOCAL_MLX_MODEL
        model, tokenizer = self._load_chat(model_name)
        normalized = _normalize_messages(messages)
        if tools:
            instruction = _tool_instruction(tools)
            if instruction:
                normalized = [{"role": "system", "content": instruction}] + normalized

        try:
            prompt = tokenizer.apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in normalized) + "\n\nASSISTANT:"

        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler
        except Exception as exc:
            raise RuntimeError("mlx_lm is not installed for direct MLX chat.") from exc

        sampler = make_sampler(temp=float(temperature or 0.0))
        gen_max_tokens = max_tokens or int(os.getenv("LOCAL_MLX_MAX_TOKENS", "4096"))

        # Build generation kwargs
        gen_kwargs: Dict[str, Any] = {
            "max_tokens": gen_max_tokens,
            "sampler": sampler,
            "verbose": False,
        }

        # TurboQuant KV cache compression
        tq_config = get_turboquant_config()
        if tq_config and tq_config.enabled and TURBOQUANT_AVAILABLE:
            try:
                from mlx_lm.models.llama import KVCache as _StockKVCache
                # Detect number of layers from model
                n_layers = 0
                if hasattr(model, "model") and hasattr(model.model, "layers"):
                    n_layers = len(model.model.layers)
                elif hasattr(model, "layers"):
                    n_layers = len(model.layers)
                if n_layers > 0:
                    prompt_cache = [TurboQuantKVCache(config=tq_config) for _ in range(n_layers)]
                    gen_kwargs["prompt_cache"] = prompt_cache
                    logger.info(
                        "TurboQuant KV compression enabled: %d-bit (%d+1) across %d layers",
                        tq_config.total_bits, tq_config.stage1_bits, n_layers,
                    )
            except Exception as exc:
                logger.warning("TurboQuant cache setup failed, using standard cache: %s", exc)

        generated = generate(
            model,
            tokenizer,
            prompt=prompt,
            **gen_kwargs,
        )
        text = generated if isinstance(generated, str) else str(generated)
        visible_text = _strip_think_artifacts(text)

        tool_calls = None
        content = visible_text
        finish_reason = "stop"
        if tools:
            parsed = _extract_json_object(text)
            if parsed and isinstance(parsed.get("tool_calls"), list):
                tool_calls = []
                for call in parsed["tool_calls"]:
                    if not isinstance(call, dict) or not call.get("name"):
                        continue
                    args = call.get("arguments", {})
                    if not isinstance(args, dict):
                        args = {}
                    tool_calls.append(
                        SimpleNamespace(
                            id=f"mlx_call_{uuid.uuid4().hex[:10]}",
                            type="function",
                            function=SimpleNamespace(
                                name=str(call["name"]),
                                arguments=json.dumps(args, ensure_ascii=False),
                            ),
                        )
                    )
                if tool_calls:
                    content = None
                    finish_reason = "tool_calls"
            if not tool_calls:
                tool_calls = _recover_xml_tool_calls(text, tools)
                if tool_calls:
                    content = None
                    finish_reason = "tool_calls"
            if not tool_calls:
                tool_calls = _recover_terminal_tool_calls(text, tools)
                if tool_calls:
                    content = None
                    finish_reason = "tool_calls"

        message = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)
        choice = SimpleNamespace(index=0, message=message, finish_reason=finish_reason)
        usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        return SimpleNamespace(choices=[choice], model=model_name, usage=usage)

    def embed_text(self, text: str, *, model_name: Optional[str] = None) -> List[float]:
        model_name = model_name or DEFAULT_LOCAL_MLX_EMBED_MODEL
        model, tokenizer = self._load_embedder(model_name)
        try:
            from mlx_embeddings import generate
        except Exception as exc:
            raise RuntimeError("mlx_embeddings is not installed for direct MLX embeddings.") from exc
        output = generate(model, tokenizer, text)
        values: Any = output
        if hasattr(output, "text_embeds") and output.text_embeds is not None:
            values = output.text_embeds
        elif hasattr(output, "pooler_output") and output.pooler_output is not None:
            values = output.pooler_output
        elif hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            values = output.last_hidden_state
        values = values.tolist() if hasattr(values, "tolist") else list(values)
        if values and isinstance(values[0], list):
            values = values[0]
        floats = [float(v) for v in values]
        norm = sum(v * v for v in floats) ** 0.5
        if norm <= 0:
            return floats
        return [v / norm for v in floats]


class _LocalCompletions:
    def __init__(self, service: LocalMLXService, model_name: str):
        self._service = service
        self._model_name = model_name

    def create(self, **kwargs) -> Any:
        model_name = kwargs.get("model") or self._model_name
        if _ensure_shared_local_mlx_server():
            payload = {
                "messages": kwargs.get("messages", []),
                "model_name": model_name,
                "tools": kwargs.get("tools"),
                "temperature": float(kwargs.get("temperature", 0.2) or 0.2),
                "max_tokens": kwargs.get("max_tokens")
                or kwargs.get("max_completion_tokens")
                or kwargs.get("max_output_tokens"),
            }
            data = _mlx_http_request("POST", "/chat_completion", payload=payload, timeout=1800)
            return _deserialize_chat_response(data)
        return self._service.chat_completion(
            kwargs.get("messages", []),
            model_name=model_name,
            tools=kwargs.get("tools"),
            temperature=float(kwargs.get("temperature", 0.2) or 0.2),
            max_tokens=kwargs.get("max_tokens")
            or kwargs.get("max_completion_tokens")
            or kwargs.get("max_output_tokens"),
        )


class _AsyncLocalCompletions:
    def __init__(self, sync_client: "LocalMLXClient"):
        self._sync_client = sync_client

    async def create(self, **kwargs) -> Any:
        return await asyncio.to_thread(self._sync_client.chat.completions.create, **kwargs)


class LocalMLXClient:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or DEFAULT_LOCAL_MLX_MODEL
        self.service = LocalMLXService.instance()
        self.chat = SimpleNamespace(completions=_LocalCompletions(self.service, self.model_name))
        self.api_key = "mlx-local"
        self.base_url = "mlx://local"


class AsyncLocalMLXClient:
    def __init__(self, model_name: Optional[str] = None):
        self._sync_client = LocalMLXClient(model_name=model_name)
        self.chat = SimpleNamespace(completions=_AsyncLocalCompletions(self._sync_client))


def build_local_mlx_client(model_name: Optional[str] = None) -> LocalMLXClient:
    return LocalMLXClient(model_name=model_name)


def build_async_local_mlx_client(model_name: Optional[str] = None) -> AsyncLocalMLXClient:
    return AsyncLocalMLXClient(model_name=model_name)


@lru_cache(maxsize=256)
def _cached_local_mlx_embed(text: str, model_name: str) -> tuple[float, ...]:
    if _ensure_shared_local_mlx_server():
        data = _mlx_http_request(
            "POST",
            "/embed",
            payload={"text": text, "model_name": model_name},
            timeout=1800,
        )
        values = data.get("embedding", []) if isinstance(data, dict) else []
        return tuple(float(v) for v in values)
    return tuple(LocalMLXService.instance().embed_text(text, model_name=model_name))


def local_mlx_embed(text: str, model_name: Optional[str] = None) -> List[float]:
    resolved_model = model_name or DEFAULT_LOCAL_MLX_EMBED_MODEL
    return list(_cached_local_mlx_embed(text, resolved_model))
