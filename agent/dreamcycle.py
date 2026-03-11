"""Idle/maintenance dream-cycle synthesis for Hermes memory graph."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from agent.auxiliary_client import get_text_auxiliary_client

logger = logging.getLogger(__name__)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    stripped = (text or "").strip()
    if not stripped:
        return None
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


class DreamCycle:
    """Bounded offline synthesis/planning pass over stored memories."""

    def __init__(self, memory_graph, *, session_id: Optional[str] = None):
        self.memory_graph = memory_graph
        self.session_id = session_id

    def _llm_plan(self, *, focus: str, packet: List[Dict[str, Any]], max_hypotheses: int) -> Dict[str, Any]:
        client, model = get_text_auxiliary_client(task="dream")
        if not client or not model:
            return {"hypotheses": [], "experiment_plans": [], "prune_candidates": []}

        prompt = (
            "You are Hermes DreamCycle, an offline memory synthesis planner.\n"
            "Given a working-memory packet, produce compact JSON only.\n"
            "Focus on:\n"
            "1. Deeper hypotheses connecting multiple memories\n"
            "2. Safe, non-destructive experiment plans to test them\n"
            "3. Prune candidates that seem stale/redundant but should be reviewed, not deleted automatically\n\n"
            f"Return JSON with keys hypotheses, experiment_plans, prune_candidates.\n"
            f"Limit hypotheses to {max_hypotheses}.\n"
            "Each hypothesis must include title, content, source_ids, predicted_terms.\n"
            "Each experiment plan must include title, content, source_ids, safe_checks.\n"
            "Each prune candidate must include node_id and reason.\n"
        )
        if focus.strip():
            prompt += f"\nCurrent focus: {focus.strip()}\n"
        prompt += "\nWorking memory packet:\n" + json.dumps(packet, ensure_ascii=False, indent=2)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.2,
            max_tokens=2200,
        )
        text = ""
        if getattr(response, "choices", None):
            text = response.choices[0].message.content or ""
        parsed = _extract_json_object(text)
        return parsed if isinstance(parsed, dict) else {"hypotheses": [], "experiment_plans": [], "prune_candidates": []}

    def run_once(self, *, focus: str = "", max_hypotheses: int = 3, task_id: Optional[str] = None) -> Dict[str, Any]:
        packet = self.memory_graph.retrieve_working_memory(
            query=focus or "active work recent hypotheses",
            session_id=self.session_id,
            task_id=task_id,
            limit=8,
        )
        synthesized_ids = self.memory_graph.synthesize_hypotheses(
            session_id=self.session_id,
            task_id=task_id,
            max_hypotheses=max_hypotheses,
        )
        llm_plan = self._llm_plan(focus=focus, packet=packet, max_hypotheses=max_hypotheses)

        created_hypotheses: List[str] = []
        for item in llm_plan.get("hypotheses", [])[:max_hypotheses]:
            if not isinstance(item, dict):
                continue
            source_ids = [str(x) for x in item.get("source_ids", []) if x]
            if not source_ids:
                continue
            hypothesis_id = self.memory_graph.add_node(
                node_type="hypothesis",
                title=str(item.get("title") or "Dream hypothesis"),
                content=str(item.get("content") or "").strip() or "Dream-cycle hypothesis",
                confidence=0.64,
                importance=0.74,
                session_id=self.session_id,
                task_id=task_id,
                source_kind="dreamcycle_hypothesis",
                metadata={
                    "source_ids": source_ids,
                    "predicted_terms": item.get("predicted_terms", []),
                    "focus": focus,
                },
                allow_duplicate=True,
            )
            for source_id in source_ids:
                self.memory_graph.add_edge(
                    src_id=source_id,
                    dst_id=hypothesis_id,
                    edge_type="derived_from",
                    metadata={"dreamcycle": True},
                )
            created_hypotheses.append(hypothesis_id)

        experiment_plan_ids: List[str] = []
        for item in llm_plan.get("experiment_plans", [])[: max_hypotheses * 2]:
            if not isinstance(item, dict):
                continue
            source_ids = [str(x) for x in item.get("source_ids", []) if x]
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            plan_id = self.memory_graph.add_node(
                node_type="plan_step",
                title=str(item.get("title") or "Dream experiment plan"),
                content=content,
                confidence=0.61,
                importance=0.66,
                session_id=self.session_id,
                task_id=task_id,
                source_kind="dreamcycle_experiment",
                metadata={
                    "source_ids": source_ids,
                    "safe_checks": item.get("safe_checks", []),
                    "focus": focus,
                },
                allow_duplicate=True,
            )
            for source_id in source_ids:
                self.memory_graph.add_edge(
                    src_id=plan_id,
                    dst_id=source_id,
                    edge_type="depends_on",
                    metadata={"dreamcycle": True},
                )
            experiment_plan_ids.append(plan_id)

        prune_candidate_ids: List[str] = []
        for item in llm_plan.get("prune_candidates", [])[:8]:
            if not isinstance(item, dict) or not item.get("node_id"):
                continue
            candidate_id = self.memory_graph.add_node(
                node_type="decision",
                title="Dream prune candidate",
                content=str(item.get("reason") or "Candidate for review pruning"),
                confidence=0.58,
                importance=0.52,
                session_id=self.session_id,
                task_id=task_id,
                source_kind="dreamcycle_prune_candidate",
                metadata={"target_node_id": str(item["node_id"]), "focus": focus},
                allow_duplicate=True,
            )
            self.memory_graph.add_edge(
                src_id=candidate_id,
                dst_id=str(item["node_id"]),
                edge_type="supports",
                metadata={"dreamcycle": True, "kind": "prune_candidate"},
            )
            prune_candidate_ids.append(candidate_id)

        return {
            "packet_size": len(packet),
            "heuristic_hypotheses": synthesized_ids,
            "dream_hypotheses": created_hypotheses,
            "experiment_plans": experiment_plan_ids,
            "prune_candidates": prune_candidate_ids,
        }
