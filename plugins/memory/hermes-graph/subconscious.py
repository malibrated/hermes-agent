"""Lightweight subconscious memory selector for Hermes."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from agent.auxiliary_client import get_text_auxiliary_client

logger = logging.getLogger(__name__)


def _extract_json_object(text: str) -> Dict[str, Any]:
    stripped = (text or "").strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


class SubconsciousSelector:
    """Select a small set of memory packet entries for deeper expansion."""

    def choose_expansions(self, *, query: str, packet: List[Dict[str, Any]], max_expand: int = 2) -> List[str]:
        if not packet or max_expand <= 0:
            return []

        client, model = get_text_auxiliary_client(task="subconscious")
        if client and model:
            try:
                prompt = (
                    "You are Hermes' subconscious memory selector. "
                    "Given the user query and a compact working-memory packet, choose at most "
                    f"{max_expand} node_id values that deserve deeper expansion. "
                    "Prefer blockers, hypotheses, and highly relevant old context only when necessary. "
                    "Return JSON only: {\"expand_node_ids\": [\"...\"]}\n\n"
                    f"User query: {query}\n\n"
                    f"Packet: {json.dumps(packet, ensure_ascii=False, indent=2)}"
                )
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500,
                )
                text = ""
                if getattr(response, "choices", None):
                    text = response.choices[0].message.content or ""
                parsed = _extract_json_object(text)
                ids = parsed.get("expand_node_ids", [])
                if isinstance(ids, list):
                    return [str(x) for x in ids[:max_expand] if x]
            except Exception as exc:
                logger.debug("Subconscious selector LLM pass failed: %s", exc)

        # Heuristic fallback: expand top-scoring blockers/hypotheses first, then strongest remaining.
        priority = {"blocker": 0, "hypothesis": 1, "decision": 2, "goal": 3, "summary": 4, "artifact": 5}
        ordered = sorted(
            packet,
            key=lambda item: (
                priority.get(str(item.get("type") or ""), 9),
                -(float(item.get("score", 0.0) or 0.0)),
            ),
        )
        return [str(item["node_id"]) for item in ordered[:max_expand] if item.get("node_id")]
