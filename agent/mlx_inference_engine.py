"""MLX inference engine with request queuing, shared prefix caching, and per-session KV management.

Architecture:
  - Single worker thread processes requests sequentially (MLX GPU serialization)
  - Shared system prompt KV cache computed once, forked per session
  - Per-session delta caches compressed via TurboQuant after threshold
  - Sessions expire after idle timeout; max session cap

Usage:
    engine = InferenceEngine.instance()
    future = engine.submit(InferenceRequest(
        session_id="agent-1",
        messages=[...],
        model_name="mlx-community/Qwen3.5-27B-...",
    ))
    result = future.result(timeout=300)  # blocks until done
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue, Empty
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import mlx.core as mx
except ImportError:
    mx = None

try:
    from mlx_vlm.turboquant import TurboQuantKVCache
    TURBOQUANT_AVAILABLE = True
except ImportError:
    TURBOQUANT_AVAILABLE = False
    TurboQuantKVCache = None


# ---------------------------------------------------------------------------
# Think-block stripping (shared helper)
# ---------------------------------------------------------------------------

def _strip_thinking_and_special_tokens(text: str) -> str:
    """Strip thinking blocks and special tokens from local model output.

    Handles:
      - Standard: <think>...</think>actual response
      - Carnice/Qwen: thinking...\n</think>\nactual response
      - Special tokens: <|im_end|>, <|endoftext|>
    """
    if not text:
        return ""
    result = text
    # Handle standard <think>...</think> blocks
    result = re.sub(r'<think>.*?</think>\s*', '', result, flags=re.DOTALL)
    # Handle Carnice-style: everything before </think> is thinking
    if '</think>' in result:
        result = result.split('</think>', 1)[-1]
    # Strip orphan think tags and special tokens
    result = re.sub(r'</?think>\s*', '', result)
    result = re.sub(r'<\|im_end\|>\s*', '', result)
    result = re.sub(r'<\|endoftext\|>\s*', '', result)
    return result.strip()


# ---------------------------------------------------------------------------
# Request / Response types
# ---------------------------------------------------------------------------

@dataclass
class InferenceRequest:
    """A queued inference request."""
    session_id: str
    messages: List[Dict[str, Any]]
    model_name: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    temperature: float = 0.2
    max_tokens: Optional[int] = None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    enqueued_at: float = 0.0  # Set by submit()
    _cancelled: threading.Event = field(default_factory=threading.Event, repr=False)

    def cancel(self) -> None:
        self._cancelled.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()


@dataclass
class InferenceResult:
    """Result from a completed inference request."""
    response: Any  # The chat completion response object
    session_id: str
    request_id: str
    cache_hit: bool = False  # True if shared prefix was reused
    queue_wait_ms: float = 0.0
    inference_ms: float = 0.0


# ---------------------------------------------------------------------------
# Session cache entry
# ---------------------------------------------------------------------------

@dataclass
class SessionCacheEntry:
    """Per-session KV cache state."""
    session_id: str
    prompt_cache: Optional[List[Any]] = None  # List of KVCache per layer
    prefix_hash: str = ""
    prefix_token_count: int = 0
    model_name: str = ""
    backend: str = ""
    last_accessed: float = field(default_factory=time.monotonic)
    total_requests: int = 0

    def touch(self) -> None:
        self.last_accessed = time.monotonic()
        self.total_requests += 1

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_accessed

    @property
    def nbytes(self) -> int:
        if not self.prompt_cache:
            return 0
        total = 0
        for layer_cache in self.prompt_cache:
            total += getattr(layer_cache, "nbytes", 0)
        return total


# ---------------------------------------------------------------------------
# Shared prefix cache
# ---------------------------------------------------------------------------

class SharedPrefixCache:
    """Frozen KV cache for the system prompt, shared across all sessions."""

    def __init__(self) -> None:
        self._frozen_state: Optional[List[Tuple[Any, Any]]] = None
        self._prompt_hash: str = ""
        self._token_count: int = 0
        self._model_name: str = ""
        self._lock = threading.Lock()

    @property
    def is_valid(self) -> bool:
        return self._frozen_state is not None

    @property
    def prompt_hash(self) -> str:
        return self._prompt_hash

    @property
    def token_count(self) -> int:
        return self._token_count

    def matches(self, prompt_hash: str, model_name: str) -> bool:
        return (
            self._frozen_state is not None
            and self._prompt_hash == prompt_hash
            and self._model_name == model_name
        )

    def build(
        self,
        model: Any,
        tokenizer: Any,
        system_messages: List[Dict[str, Any]],
        model_name: str,
    ) -> None:
        """Encode the system prompt and freeze the KV cache."""
        with self._lock:
            prompt_hash = _hash_messages(system_messages)
            if self.matches(prompt_hash, model_name):
                return

            logger.info("Building shared prefix cache for model=%s", model_name)

            prompt_cache = _make_cache_for_model(model)

            try:
                text = tokenizer.apply_chat_template(
                    system_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                text = "\n\n".join(
                    f"{m.get('role', 'system').upper()}: {m.get('content', '')}"
                    for m in system_messages
                )

            _tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
            tokens = _tok.encode(text)
            if not tokens:
                self._frozen_state = None
                return

            token_array = mx.array(tokens)
            model(token_array[None], cache=prompt_cache)
            mx.eval(*[c.state for c in prompt_cache if hasattr(c, "state")])

            self._frozen_state = []
            self._token_count = len(tokens)
            for layer_cache in prompt_cache:
                if hasattr(layer_cache, "state"):
                    keys, values = layer_cache.state
                    self._frozen_state.append((mx.array(keys), mx.array(values)))
                else:
                    self._frozen_state.append(None)

            self._prompt_hash = prompt_hash
            self._model_name = model_name
            logger.info(
                "Shared prefix cache built: %d tokens, %d layers",
                self._token_count, len(self._frozen_state),
            )

    def fork(self, use_turboquant: bool = False, tq_bits: float = 4.0, tq_seed: int = 0) -> List[Any]:
        """Create a new per-session cache forked from the frozen prefix."""
        if self._frozen_state is None:
            raise RuntimeError("SharedPrefixCache not built yet")

        forked: List[Any] = []
        for layer_state in self._frozen_state:
            if layer_state is None:
                if use_turboquant and TURBOQUANT_AVAILABLE:
                    forked.append(TurboQuantKVCache(bits=tq_bits, seed=tq_seed))
                else:
                    from mlx_lm.models.llama import KVCache
                    forked.append(KVCache())
                continue

            keys, values = layer_state

            if use_turboquant and TURBOQUANT_AVAILABLE:
                from mlx_lm.models.llama import KVCache
                stock = KVCache()
                stock.state = (mx.array(keys), mx.array(values))
                cache = TurboQuantKVCache.from_cache(stock, bits=tq_bits, seed=tq_seed)
            else:
                from mlx_lm.models.llama import KVCache
                cache = KVCache()
                cache.state = (mx.array(keys), mx.array(values))

            forked.append(cache)

        return forked


# ---------------------------------------------------------------------------
# Inference engine (singleton)
# ---------------------------------------------------------------------------

DEFAULT_MAX_SESSIONS = int(os.getenv("MLX_MAX_SESSIONS", "6"))
DEFAULT_IDLE_TIMEOUT = float(os.getenv("MLX_SESSION_IDLE_TIMEOUT", "600"))
DEFAULT_MAX_QUEUE = int(os.getenv("MLX_MAX_QUEUE_SIZE", "16"))


class InferenceEngine:
    """Singleton MLX inference engine with queuing and session caching."""

    _instance: Optional["InferenceEngine"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        from agent.local_mlx import LocalMLXService
        self._service = LocalMLXService.instance()

        self._queue: Queue[Tuple[InferenceRequest, Future]] = Queue(
            maxsize=DEFAULT_MAX_QUEUE
        )
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

        self._sessions: OrderedDict[str, SessionCacheEntry] = OrderedDict()
        self._sessions_lock = threading.Lock()
        self._max_sessions = DEFAULT_MAX_SESSIONS
        self._idle_timeout = DEFAULT_IDLE_TIMEOUT

        self._prefix_cache = SharedPrefixCache()

        from agent.local_mlx import get_turboquant_config
        self._tq_config = get_turboquant_config()
        self._use_turboquant = (
            TURBOQUANT_AVAILABLE
            and self._tq_config is not None
        )

        self._start_worker()

    @classmethod
    def instance(cls) -> "InferenceEngine":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def submit(self, request: InferenceRequest) -> Future:
        """Submit an inference request to the queue."""
        future: Future = Future()
        request.enqueued_at = time.monotonic()
        future._mlx_request = request  # attach for cancellation
        self._queue.put((request, future), block=False)
        return future

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def active_sessions(self) -> int:
        with self._sessions_lock:
            return len(self._sessions)

    def get_stats(self) -> Dict[str, Any]:
        with self._sessions_lock:
            sessions_info = []
            for sid, entry in self._sessions.items():
                sessions_info.append({
                    "session_id": entry.session_id,
                    "cache_key": sid,
                    "model": entry.model_name,
                    "backend": entry.backend,
                    "idle_seconds": round(entry.idle_seconds, 1),
                    "total_requests": entry.total_requests,
                    "prefix_tokens": entry.prefix_token_count,
                    "cache_bytes": entry.nbytes,
                })

        return {
            "queue_depth": self.queue_depth,
            "active_sessions": self.active_sessions,
            "max_sessions": self._max_sessions,
            "shared_prefix_valid": self._prefix_cache.is_valid,
            "shared_prefix_tokens": self._prefix_cache.token_count,
            "turboquant_enabled": self._use_turboquant,
            "sessions": sessions_info,
        }

    def shutdown(self) -> None:
        self._shutdown.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10.0)
        with self._sessions_lock:
            self._sessions.clear()

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _start_worker(self) -> None:
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="mlx-inference-worker"
        )
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        logger.info("MLX inference worker started")
        while not self._shutdown.is_set():
            self._expire_idle_sessions()

            try:
                request, future = self._queue.get(timeout=5.0)
            except Empty:
                continue

            # Skip cancelled requests before spending GPU time
            if request.is_cancelled:
                future.set_exception(RuntimeError("Request cancelled"))
                self._queue.task_done()
                continue

            try:
                result = self._process_request(request)
                # Don't persist result for cancelled requests
                if request.is_cancelled:
                    future.set_exception(RuntimeError("Request cancelled during processing"))
                else:
                    future.set_result(result)
            except Exception as exc:
                logger.exception("Inference request %s failed", request.request_id)
                future.set_exception(exc)
            finally:
                self._queue.task_done()

        logger.info("MLX inference worker stopped")

    # ------------------------------------------------------------------
    # Request processing
    # ------------------------------------------------------------------

    def _process_request(self, request: InferenceRequest) -> InferenceResult:
        t_start = time.monotonic()
        queue_wait_ms = (t_start - request.enqueued_at) * 1000 if request.enqueued_at else 0

        model_name = request.model_name or os.getenv(
            "LOCAL_MLX_MODEL",
            "mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit",
        )

        # Load model (returns 3-tuple: model, tokenizer_or_processor, backend)
        loaded = self._service._load_chat(model_name)
        model, tokenizer_or_processor, backend = loaded

        # Split messages into system prefix and conversation
        system_msgs, conversation_msgs = _split_system_messages(request.messages)

        # Skip prefix cache for models with custom cache requirements (e.g. hybrid SSM)
        has_custom_cache = hasattr(model, "make_cache")
        cache_hit = False
        current_prefix_hash = _hash_messages(system_msgs) if system_msgs else ""

        if system_msgs and not has_custom_cache:
            if not self._prefix_cache.matches(current_prefix_hash, model_name):
                self._prefix_cache.build(model, tokenizer_or_processor, system_msgs, model_name)

        # Get or create session cache (keyed by session + backend + model)
        cache_key = f"{request.session_id}:{backend}:{model_name}"
        session = self._get_or_create_session(cache_key, request.session_id, model_name, backend)
        session.touch()

        # Validate session cache staleness
        if session.prompt_cache is not None:
            stale = (
                (current_prefix_hash and session.prefix_hash != current_prefix_hash)
                or session.model_name != model_name
                or session.backend != backend
            )
            if stale:
                logger.info(
                    "Invalidating stale session cache %s: prefix %s->%s model %s->%s",
                    cache_key, session.prefix_hash[:8], current_prefix_hash[:8],
                    session.model_name, model_name,
                )
                session.prompt_cache = None
                session.prefix_hash = ""
                session.prefix_token_count = 0

        from agent.local_mlx import (
            _normalize_messages, _extract_json_object,
            _recover_xml_tool_calls, _recover_terminal_tool_calls,
        )

        normalized = _normalize_messages(request.messages)

        try:
            prompt_text = tokenizer_or_processor.apply_chat_template(
                normalized, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            prompt_text = "\n\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in normalized
            ) + "\n\nASSISTANT:"

        max_tokens = request.max_tokens or int(
            os.getenv("LOCAL_MLX_MAX_TOKENS", "8192")
        )

        # Generate using the appropriate backend
        if backend == "mlx_vlm":
            from mlx_vlm import generate as vlm_generate
            from agent.local_mlx import get_turboquant_config

            gen_kwargs: Dict[str, Any] = {
                "max_tokens": max_tokens,
                "temperature": float(request.temperature or 0.0),
                "verbose": False,
            }
            tq_config = get_turboquant_config()
            if tq_config:
                gen_kwargs["kv_bits"] = tq_config["bits"]

            logger.info(
                "vlm_generate: model=%s max_tokens=%s prompt_len=%d",
                model_name, gen_kwargs.get("max_tokens"), len(prompt_text),
            )
            result = vlm_generate(
                model, tokenizer_or_processor,
                prompt=prompt_text, **gen_kwargs,
            )
            text = result.text if hasattr(result, "text") else str(result)
            logger.info(
                "vlm_generate result: %d chars, starts_with=%r",
                len(text), text[:100] if text else "",
            )
        else:
            from mlx_lm import generate as lm_generate
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=float(request.temperature or 0.0))
            gen_kwargs: Dict[str, Any] = {
                "max_tokens": max_tokens,
                "sampler": sampler,
                "verbose": False,
            }

            if has_custom_cache:
                if self._use_turboquant and self._tq_config:
                    gen_kwargs["kv_bits"] = int(self._tq_config.get("bits", 4))
                    logger.info(
                        "TurboQuant via mlx_lm kv_bits=%d for hybrid model",
                        gen_kwargs["kv_bits"],
                    )
            elif session.prompt_cache is not None:
                gen_kwargs["prompt_cache"] = session.prompt_cache
                cache_hit = True
            elif self._prefix_cache.is_valid:
                try:
                    session.prompt_cache = self._prefix_cache.fork(
                        use_turboquant=self._use_turboquant,
                        tq_bits=self._tq_config.get("bits", 4.0) if self._tq_config else 4.0,
                        tq_seed=self._tq_config.get("seed", 0) if self._tq_config else 0,
                    )
                    session.prefix_token_count = self._prefix_cache.token_count
                    session.prefix_hash = self._prefix_cache.prompt_hash
                    session.model_name = model_name
                    session.backend = backend
                    gen_kwargs["prompt_cache"] = session.prompt_cache
                    cache_hit = True
                except Exception as exc:
                    logger.warning("Failed to fork prefix cache: %s", exc)

            logger.info(
                "lm_generate: model=%s max_tokens=%s prompt_len=%d",
                model_name, gen_kwargs.get("max_tokens"), len(prompt_text),
            )
            generated = lm_generate(
                model, tokenizer_or_processor,
                prompt=prompt_text, **gen_kwargs,
            )
            text = generated if isinstance(generated, str) else str(generated)
            logger.info(
                "lm_generate result: %d chars, starts_with=%r",
                len(text), text[:200] if text else "",
            )

        # --- FIX 1: Strip thinking FIRST, then parse tool calls ---
        visible_text = _strip_thinking_and_special_tokens(text)

        # Parse tool calls from CLEANED text only
        tool_calls = None
        content = visible_text
        finish_reason = "stop"

        if request.tools:
            # 1. Gemma 4 native format
            gemma_calls = re.findall(
                r'<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>',
                visible_text, re.DOTALL
            )
            if gemma_calls:
                tool_calls = []
                for name, args_str in gemma_calls:
                    try:
                        args = json.loads("{" + args_str + "}") if not args_str.startswith("{") else json.loads(args_str)
                    except (json.JSONDecodeError, Exception):
                        args = {}
                        for pair in args_str.split(","):
                            if ":" in pair:
                                k, v = pair.split(":", 1)
                                args[k.strip().strip("'")] = v.strip().strip("'")
                    tool_calls.append(
                        SimpleNamespace(
                            id=f"mlx_call_{uuid.uuid4().hex[:10]}",
                            type="function",
                            function=SimpleNamespace(
                                name=str(name),
                                arguments=json.dumps(args, ensure_ascii=False),
                            ),
                        )
                    )
                if tool_calls:
                    content = None
                    finish_reason = "tool_calls"

            # 2. JSON format
            if not tool_calls:
                parsed = _extract_json_object(visible_text)
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

            # 3. XML format
            if not tool_calls:
                tool_calls = _recover_xml_tool_calls(visible_text, request.tools)
                if tool_calls:
                    content = None
                    finish_reason = "tool_calls"

            # 4. Terminal code blocks
            if not tool_calls:
                tool_calls = _recover_terminal_tool_calls(visible_text, request.tools)
                if tool_calls:
                    content = None
                    finish_reason = "tool_calls"

        logger.info(
            "Response: content=%r tool_calls=%s finish=%s",
            (content[:200] if content else None),
            len(tool_calls) if tool_calls else 0,
            finish_reason,
        )
        message = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)
        choice = SimpleNamespace(index=0, message=message, finish_reason=finish_reason)
        usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        response = SimpleNamespace(choices=[choice], model=model_name, usage=usage)

        t_end = time.monotonic()
        return InferenceResult(
            response=response,
            session_id=request.session_id,
            request_id=request.request_id,
            cache_hit=cache_hit,
            queue_wait_ms=queue_wait_ms,
            inference_ms=(t_end - t_start) * 1000,
        )

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _get_or_create_session(
        self, cache_key: str, session_id: str, model_name: str, backend: str
    ) -> SessionCacheEntry:
        with self._sessions_lock:
            if cache_key in self._sessions:
                entry = self._sessions[cache_key]
                self._sessions.move_to_end(cache_key)
                return entry

            while len(self._sessions) >= self._max_sessions:
                oldest_id, oldest = next(iter(self._sessions.items()))
                logger.info(
                    "Session cap reached, expiring oldest session %s (idle %.0fs)",
                    oldest_id, oldest.idle_seconds,
                )
                del self._sessions[oldest_id]

            entry = SessionCacheEntry(
                session_id=session_id,
                model_name=model_name,
                backend=backend,
            )
            self._sessions[cache_key] = entry
            return entry

    def _expire_idle_sessions(self) -> None:
        now = time.monotonic()
        with self._sessions_lock:
            expired = [
                sid for sid, entry in self._sessions.items()
                if (now - entry.last_accessed) > self._idle_timeout
            ]
            for sid in expired:
                logger.info("Expiring idle session %s", sid)
                del self._sessions[sid]

    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate all cache entries for a session (any backend/model)."""
        with self._sessions_lock:
            to_remove = [k for k in self._sessions if k.startswith(f"{session_id}:")]
            for k in to_remove:
                del self._sessions[k]
            return len(to_remove) > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_messages(messages: List[Dict[str, Any]]) -> str:
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _split_system_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    system: List[Dict[str, Any]] = []
    rest: List[Dict[str, Any]] = []
    in_system = True
    for msg in messages:
        if in_system and msg.get("role") == "system":
            system.append(msg)
        else:
            in_system = False
            rest.append(msg)
    return system, rest


def _make_cache_for_model(model: Any) -> List[Any]:
    """Create appropriate cache list for the model.

    Handles hybrid architectures (e.g. Qwen 3.5 attention+SSM) by checking
    for make_cache() on the model or its language_model wrapper, which returns
    the correct mix of KVCache (attention) and ArraysCache (SSM) per layer.
    """
    if hasattr(model, "make_cache"):
        return model.make_cache()
    if hasattr(model, "language_model") and hasattr(model.language_model, "make_cache"):
        return model.language_model.make_cache()
    from mlx_lm.models.llama import KVCache
    num_layers = len(model.layers) if hasattr(model, "layers") else 0
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
    return [KVCache() for _ in range(num_layers)]
