"""Shared loopback server for local MLX inference.

v2: Routes chat completions through InferenceEngine with request queuing,
shared system prompt prefix caching, and per-session KV cache management.
Embedding requests bypass the queue (lightweight, no KV cache).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Full
from typing import Any, Dict

from agent.local_mlx import (
    DEFAULT_LOCAL_MLX_PORT,
    LocalMLXService,
    get_turboquant_config,
    TURBOQUANT_AVAILABLE,
)
from agent.mlx_inference_engine import InferenceEngine, InferenceRequest

logger = logging.getLogger(__name__)

# Embedding requests are lightweight — serialize with a simple lock
_EMBED_LOCK = threading.Lock()
_EMBED_SERVICE = LocalMLXService.instance()

# Chat completions go through the engine
_ENGINE: InferenceEngine | None = None


def _get_engine() -> InferenceEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = InferenceEngine.instance()
    return _ENGINE


def _tool_call_to_dict(tool_call: Any) -> Dict[str, Any]:
    function = getattr(tool_call, "function", None)
    return {
        "id": getattr(tool_call, "id", ""),
        "type": getattr(tool_call, "type", "function"),
        "function": {
            "name": getattr(function, "name", ""),
            "arguments": getattr(function, "arguments", "{}"),
        },
    }


def _chat_response_to_dict(response: Any) -> Dict[str, Any]:
    import time
    choices = []
    for choice in getattr(response, "choices", []) or []:
        message = getattr(choice, "message", None)
        tool_calls = [
            _tool_call_to_dict(tc)
            for tc in (getattr(message, "tool_calls", None) or [])
        ]
        choices.append(
            {
                "index": getattr(choice, "index", 0),
                "finish_reason": getattr(choice, "finish_reason", "stop"),
                "message": {
                    "role": getattr(message, "role", "assistant"),
                    "content": getattr(message, "content", None),
                    **({"tool_calls": tool_calls} if tool_calls else {}),
                },
            }
        )
    usage = getattr(response, "usage", None)
    return {
        "id": f"chatcmpl-local-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": getattr(response, "model", ""),
        "choices": choices,
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        },
    }


class _Handler(BaseHTTPRequestHandler):
    server_version = "HermesLocalMLX/2.0"

    def log_message(self, format: str, *args) -> None:
        logger.debug("local-mlx-server: " + format, *args)

    def _write_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length > 0 else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    # ------------------------------------------------------------------
    # GET endpoints
    # ------------------------------------------------------------------

    def do_GET(self) -> None:
        if self.path == "/health":
            engine = _get_engine()
            tq_config = get_turboquant_config()
            tq_info = {}
            if TURBOQUANT_AVAILABLE and tq_config:
                tq_info = {
                    "turboquant_available": True,
                    "turboquant_enabled": tq_config.enabled,
                    "turboquant_bits": f"{tq_config.stage1_bits}+1",
                }
            else:
                tq_info = {"turboquant_available": TURBOQUANT_AVAILABLE}
            self._write_json(200, {
                "status": "ok",
                "queue_depth": engine.queue_depth,
                "active_sessions": engine.active_sessions,
                **tq_info,
            })
            return

        if self.path == "/stats":
            engine = _get_engine()
            self._write_json(200, engine.get_stats())
            return

        if self.path.startswith("/sessions"):
            engine = _get_engine()
            stats = engine.get_stats()
            self._write_json(200, {"sessions": stats.get("sessions", [])})
            return

        # OpenAI-compatible /v1/models endpoint
        if self.path in ("/v1/models", "/models"):
            import os
            model_id = os.getenv("LOCAL_MLX_MODEL", "local-mlx")
            self._write_json(200, {
                "object": "list",
                "data": [{"id": model_id, "object": "model", "owned_by": "local-mlx"}],
            })
            return

        self._write_json(404, {"error": f"unknown path: {self.path}"})

    # ------------------------------------------------------------------
    # POST endpoints
    # ------------------------------------------------------------------

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
        except Exception as exc:
            self._write_json(400, {"error": f"invalid json: {exc}"})
            return

        try:
            # OpenAI-compatible routes (used by Hermes custom provider via OpenAI SDK)
            if self.path in ("/v1/chat/completions", "/chat/completions", "/chat_completion"):
                self._handle_chat_completion(payload)
                return

            if self.path in ("/v1/embeddings", "/embed"):
                self._handle_embed(payload)
                return

            if self.path == "/invalidate_session":
                session_id = payload.get("session_id", "")
                if not session_id:
                    self._write_json(400, {"error": "session_id required"})
                    return
                engine = _get_engine()
                removed = engine.invalidate_session(session_id)
                self._write_json(200, {"invalidated": removed, "session_id": session_id})
                return

            self._write_json(404, {"error": f"unknown path: {self.path}"})
        except Exception as exc:
            logger.exception("Local MLX server request failed")
            self._write_json(500, {"error": str(exc)})

    def _handle_chat_completion(self, payload: Dict[str, Any]) -> None:
        """Submit a chat completion to the inference engine queue."""
        engine = _get_engine()

        # Extract session_id — default to a per-request unique ID if not provided
        session_id = payload.get("session_id", "")
        if not session_id:
            # Check header too
            session_id = self.headers.get("X-Session-Id", "")
        if not session_id:
            import uuid
            session_id = f"anon-{uuid.uuid4().hex[:8]}"

        request = InferenceRequest(
            session_id=session_id,
            messages=payload.get("messages", []),
            model_name=payload.get("model") or payload.get("model_name"),
            tools=payload.get("tools"),
            temperature=float(payload.get("temperature", 0.2) or 0.2),
            max_tokens=payload.get("max_tokens") or payload.get("max_completion_tokens"),
        )

        try:
            future = engine.submit(request)
        except Full:
            self._write_json(503, {
                "error": "inference queue full",
                "queue_depth": engine.queue_depth,
                "retry_after_seconds": 5,
            })
            return

        # Block until the result is ready (timeout matches client expectation)
        timeout = float(payload.get("timeout", 1800))
        try:
            result = future.result(timeout=timeout)
        except TimeoutError:
            self._write_json(504, {
                "error": "inference timeout",
                "session_id": session_id,
                "request_id": request.request_id,
            })
            return
        except Exception as exc:
            self._write_json(500, {"error": str(exc)})
            return

        response_dict = _chat_response_to_dict(result.response)
        response_dict["_meta"] = {
            "session_id": result.session_id,
            "request_id": result.request_id,
            "cache_hit": result.cache_hit,
            "queue_wait_ms": round(result.queue_wait_ms, 1),
            "inference_ms": round(result.inference_ms, 1),
        }
        self._write_json(200, response_dict)

    def _handle_embed(self, payload: Dict[str, Any]) -> None:
        """Embedding requests bypass the queue — they're lightweight."""
        with _EMBED_LOCK:
            embedding = _EMBED_SERVICE.embed_text(
                str(payload.get("text", "")),
                model_name=payload.get("model_name"),
            )
        self._write_json(200, {"embedding": embedding})


# ------------------------------------------------------------------
# DELETE endpoint for session management
# ------------------------------------------------------------------

    def do_DELETE(self) -> None:
        if self.path.startswith("/sessions/"):
            session_id = self.path.split("/sessions/", 1)[1]
            if session_id:
                engine = _get_engine()
                removed = engine.invalidate_session(session_id)
                self._write_json(200, {"invalidated": removed, "session_id": session_id})
                return
        self._write_json(404, {"error": f"unknown path: {self.path}"})


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOCAL_MLX_SERVER_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    port = int(os.getenv("LOCAL_MLX_PORT", str(DEFAULT_LOCAL_MLX_PORT)))
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    logger.info("Local MLX inference server v2 listening on 127.0.0.1:%s", port)
    logger.info(
        "Config: max_sessions=%s, idle_timeout=%ss, max_queue=%s",
        os.getenv("MLX_MAX_SESSIONS", "24"),
        os.getenv("MLX_SESSION_IDLE_TIMEOUT", "1800"),
        os.getenv("MLX_MAX_QUEUE_SIZE", "64"),
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        engine = _get_engine()
        engine.shutdown()
        server.shutdown()


if __name__ == "__main__":
    main()
