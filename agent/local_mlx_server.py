"""Shared loopback server for local MLX inference."""

from __future__ import annotations

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

from agent.local_mlx import DEFAULT_LOCAL_MLX_PORT, LocalMLXService

logger = logging.getLogger(__name__)

_REQUEST_LOCK = threading.Lock()
_SERVICE = LocalMLXService.instance()


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
    choices = []
    for choice in getattr(response, "choices", []) or []:
        message = getattr(choice, "message", None)
        choices.append(
            {
                "index": getattr(choice, "index", 0),
                "finish_reason": getattr(choice, "finish_reason", "stop"),
                "message": {
                    "role": getattr(message, "role", "assistant"),
                    "content": getattr(message, "content", None),
                    "tool_calls": [
                        _tool_call_to_dict(tool_call)
                        for tool_call in (getattr(message, "tool_calls", None) or [])
                    ],
                },
            }
        )
    usage = getattr(response, "usage", None)
    return {
        "choices": choices,
        "model": getattr(response, "model", ""),
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        },
    }


class _Handler(BaseHTTPRequestHandler):
    server_version = "HermesLocalMLX/1.0"

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

    def do_GET(self) -> None:
        if self.path == "/health":
            self._write_json(200, {"status": "ok"})
            return
        self._write_json(404, {"error": f"unknown path: {self.path}"})

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
        except Exception as exc:
            self._write_json(400, {"error": f"invalid json: {exc}"})
            return

        try:
            if self.path == "/chat_completion":
                with _REQUEST_LOCK:
                    response = _SERVICE.chat_completion(
                        payload.get("messages", []),
                        model_name=payload.get("model_name"),
                        tools=payload.get("tools"),
                        temperature=float(payload.get("temperature", 0.2) or 0.2),
                        max_tokens=payload.get("max_tokens"),
                    )
                self._write_json(200, _chat_response_to_dict(response))
                return

            if self.path == "/embed":
                with _REQUEST_LOCK:
                    embedding = _SERVICE.embed_text(
                        str(payload.get("text", "")),
                        model_name=payload.get("model_name"),
                    )
                self._write_json(200, {"embedding": embedding})
                return

            self._write_json(404, {"error": f"unknown path: {self.path}"})
        except Exception as exc:
            logger.exception("Local MLX server request failed")
            self._write_json(500, {"error": str(exc)})


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOCAL_MLX_SERVER_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    port = int(os.getenv("LOCAL_MLX_PORT", str(DEFAULT_LOCAL_MLX_PORT)))
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    logger.info("Local MLX server listening on 127.0.0.1:%s", port)
    server.serve_forever()


if __name__ == "__main__":
    main()
