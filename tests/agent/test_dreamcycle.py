from pathlib import Path

from agent.dreamcycle import DreamCycle
from agent.memory_graph import MemoryGraphStore


def test_dreamcycle_runs_in_heuristic_mode_without_aux_llm(tmp_path, monkeypatch):
    store = MemoryGraphStore(db_path=Path(tmp_path / "memory_graph.db"))
    vectors = {
        "Observation\nVault migration changed the workspace path layout": [1.0, 0.0, 0.0],
        "Artifact\nObsidian lookup broke after the workspace path layout changed": [0.7, 0.3, 0.0],
    }

    def fake_embed(text: str):
        return vectors[text.strip()]

    store._embedder = fake_embed
    store._embedder_model = "test-embedder"

    store.add_node(
        node_type="observation",
        title="Observation",
        content="Vault migration changed the workspace path layout",
        session_id="s1",
        task_id="t1",
        allow_duplicate=True,
    )
    store.add_node(
        node_type="artifact",
        title="Artifact",
        content="Obsidian lookup broke after the workspace path layout changed",
        session_id="s1",
        task_id="t2",
        allow_duplicate=True,
    )

    monkeypatch.setattr("agent.dreamcycle.get_text_auxiliary_client", lambda task="": (None, None))

    result = DreamCycle(store, session_id="s1").run_once(focus="vault path issues", max_hypotheses=2)

    assert result["packet_size"] >= 1
    assert result["heuristic_hypotheses"]
