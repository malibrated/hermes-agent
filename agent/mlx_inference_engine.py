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
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue, Empty
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
    # Hash of the messages that were encoded into the cache, so we can detect
    # whether the client's conversation prefix still matches.
    prefix_hash: str = ""
    prefix_token_count: int = 0
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
    """Frozen KV cache for the system prompt, shared across all sessions.

    The system prompt (tools, identity, guidance) is encoded once into a
    KV cache.  Per-session caches fork from this frozen snapshot via
    deep copy of the MLX arrays, skipping re-encoding the prompt.
    """

    def __init__(self) -> None:
        self._frozen_state: Optional[List[Tuple[Any, Any]]] = None  # [(keys, values), ...] per layer
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
                return  # Already up to date

            logger.info("Building shared prefix cache for model=%s", model_name)

            from mlx_lm.models.cache import make_prompt_cache

            prompt_cache = _make_cache_for_model(model)

            # Tokenize just the system messages
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

            # Extract the actual tokenizer from processor objects (e.g. Gemma4Processor)
            _tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
            tokens = _tok.encode(text)
            if not tokens:
                self._frozen_state = None
                return

            token_array = mx.array(tokens)

            # Run the model forward to populate the cache
            model(token_array[None], cache=prompt_cache)
            mx.eval(*[c.state for c in prompt_cache if hasattr(c, "state")])

            # Snapshot the state
            self._frozen_state = []
            self._token_count = len(tokens)
            for layer_cache in prompt_cache:
                if hasattr(layer_cache, "state"):
                    keys, values = layer_cache.state
                    # Deep copy so the frozen state is immutable
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
        """Create a new per-session cache forked from the frozen prefix.

        Returns a list of KVCache objects (one per layer) pre-populated
        with the system prompt's KV state.
        """
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
                # Create a stock cache with the prefix state, then convert
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

# Defaults
DEFAULT_MAX_SESSIONS = int(os.getenv("MLX_MAX_SESSIONS", "24"))
DEFAULT_IDLE_TIMEOUT = float(os.getenv("MLX_SESSION_IDLE_TIMEOUT", "1800"))  # 30 min
DEFAULT_MAX_QUEUE = int(os.getenv("MLX_MAX_QUEUE_SIZE", "64"))


class InferenceEngine:
    """Singleton MLX inference engine with queuing and session caching."""

    _instance: Optional["InferenceEngine"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        from agent.local_mlx import LocalMLXService
        self._service = LocalMLXService.instance()

        # Request queue
        self._queue: Queue[Tuple[InferenceRequest, Future]] = Queue(
            maxsize=DEFAULT_MAX_QUEUE
        )
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

        # Session caches
        self._sessions: OrderedDict[str, SessionCacheEntry] = OrderedDict()
        self._sessions_lock = threading.Lock()
        self._max_sessions = DEFAULT_MAX_SESSIONS
        self._idle_timeout = DEFAULT_IDLE_TIMEOUT

        # Shared prefix
        self._prefix_cache = SharedPrefixCache()

        # TurboQuant config
        from agent.local_mlx import get_turboquant_config
        self._tq_config = get_turboquant_config()  # dict or None
        self._use_turboquant = (
            TURBOQUANT_AVAILABLE
            and self._tq_config is not None
        )

        # Start worker
        self._start_worker()

    @classmethod
    def instance(cls) -> "InferenceEngine":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def submit(self, request: InferenceRequest) -> Future:
        """Submit an inference request to the queue.

        Returns a Future that resolves to an InferenceResult.
        Raises queue.Full if the queue is at capacity.
        """
        future: Future = Future()
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
        """Return engine statistics."""
        with self._sessions_lock:
            sessions_info = []
            for sid, entry in self._sessions.items():
                sessions_info.append({
                    "session_id": sid,
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
        """Stop the worker thread and clean up."""
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
        """Process requests one at a time from the queue."""
        logger.info("MLX inference worker started")
        while not self._shutdown.is_set():
            # Periodic idle session cleanup
            self._expire_idle_sessions()

            try:
                request, future = self._queue.get(timeout=5.0)
            except Empty:
                continue

            t_dequeue = time.monotonic()
            try:
                result = self._process_request(request, t_dequeue)
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

    def _process_request(
        self, request: InferenceRequest, t_dequeue: float
    ) -> InferenceResult:
        """Process a single inference request with session caching."""
        t_start = time.monotonic()
        queue_wait_ms = (t_start - t_dequeue) * 1000 if t_dequeue else 0

        model_name = request.model_name or os.getenv(
            "LOCAL_MLX_MODEL",
            "mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit",
        )

        # Load model (returns 3-tuple: model, tokenizer_or_processor, backend)
        loaded = self._service._load_chat(model_name)
        model, tokenizer_or_processor, backend = loaded

        # Split messages into system prefix and conversation
        system_msgs, conversation_msgs = _split_system_messages(request.messages)

        # Ensure shared prefix cache is up to date
        cache_hit = False
        if system_msgs:
            prefix_hash = _hash_messages(system_msgs)
            if not self._prefix_cache.matches(prefix_hash, model_name):
                self._prefix_cache.build(model, tokenizer_or_processor, system_msgs, model_name)

        # Get or create session cache
        session = self._get_or_create_session(request.session_id, model_name)
        session.touch()

        from agent.local_mlx import (
            _normalize_messages, _tool_instruction, _strip_think_artifacts,
            _extract_json_object, _recover_xml_tool_calls,
            _recover_terminal_tool_calls,
        )

        normalized = _normalize_messages(request.messages)
        if request.tools:
            instruction = _tool_instruction(request.tools)
            if instruction:
                normalized = [{"role": "system", "content": instruction}] + normalized

        try:
            prompt_text = tokenizer_or_processor.apply_chat_template(
                normalized, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            prompt_text = "\n\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in normalized
            ) + "\n\nASSISTANT:"

        max_tokens = request.max_tokens or int(
            os.getenv("LOCAL_MLX_MAX_TOKENS", "4096")
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

            result = vlm_generate(
                model, tokenizer_or_processor,
                prompt=prompt_text, **gen_kwargs,
            )
            text = result.text if hasattr(result, "text") else str(result)
        else:
            from mlx_lm import generate as lm_generate
            from mlx_lm.sample_utils import make_sampler

            sampler = make_sampler(temp=float(request.temperature or 0.0))
            gen_kwargs: Dict[str, Any] = {
                "max_tokens": max_tokens,
                "sampler": sampler,
                "verbose": False,
            }

            # Session cache (only for mlx_lm path — mlx_vlm manages its own)
            if session.prompt_cache is not None:
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
                    gen_kwargs["prompt_cache"] = session.prompt_cache
                    cache_hit = True
                except Exception as exc:
                    logger.warning("Failed to fork prefix cache: %s", exc)

            generated = lm_generate(
                model, tokenizer_or_processor,
                prompt=prompt_text, **gen_kwargs,
            )
            text = generated if isinstance(generated, str) else str(generated)

        visible_text = _strip_think_artifacts(text)

        # Parse tool calls (same logic as LocalMLXService.chat_completion)
        from types import SimpleNamespace
        tool_calls = None
        content = visible_text
        finish_reason = "stop"

        if request.tools:
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
                tool_calls = _recover_xml_tool_calls(text, request.tools)
                if tool_calls:
                    content = None
                    finish_reason = "tool_calls"
            if not tool_calls:
                tool_calls = _recover_terminal_tool_calls(text, request.tools)
                if tool_calls:
                    content = None
                    finish_reason = "tool_calls"

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
        self, session_id: str, model_name: str
    ) -> SessionCacheEntry:
        with self._sessions_lock:
            if session_id in self._sessions:
                entry = self._sessions[session_id]
                # Move to end (most recently used)
                self._sessions.move_to_end(session_id)
                return entry

            # Enforce max sessions — expire oldest if at cap
            while len(self._sessions) >= self._max_sessions:
                oldest_id, oldest = next(iter(self._sessions.items()))
                logger.info(
                    "Session cap reached, expiring oldest session %s (idle %.0fs)",
                    oldest_id, oldest.idle_seconds,
                )
                del self._sessions[oldest_id]

            entry = SessionCacheEntry(session_id=session_id)
            self._sessions[session_id] = entry
            return entry

    def _expire_idle_sessions(self) -> None:
        """Remove sessions that have been idle beyond the timeout."""
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
        """Explicitly invalidate a session's cache."""
        with self._sessions_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_messages(messages: List[Dict[str, Any]]) -> str:
    """Deterministic hash of a message list for cache key comparison."""
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _split_system_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split messages into leading system messages and the rest."""
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
    """Create appropriate cache list for the model."""
    if hasattr(model, "make_cache"):
        return model.make_cache()
    from mlx_lm.models.llama import KVCache
    num_layers = len(model.layers) if hasattr(model, "layers") else 0
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
    return [KVCache() for _ in range(num_layers)]
