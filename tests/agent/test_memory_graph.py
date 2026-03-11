from pathlib import Path

from agent.memory_graph import MemoryGraphStore


def test_memory_graph_ingest_and_retrieve(tmp_path):
    store = MemoryGraphStore(db_path=Path(tmp_path / "memory_graph.db"))
    store.ingest_memory_entry(
        target="memory",
        content="The repo uses pytest and stores sessions in SQLite.",
        session_id="s1",
    )
    store.record_session_goal(session_id="s1", content="Implement structured memory retrieval")

    rows = store.retrieve_frontier(query="SQLite", session_id="s1", limit=5)

    assert rows
    assert any("SQLite" in (row.get("content") or "") for row in rows)


def test_memory_graph_dedupes_exact_duplicate_entries(tmp_path):
    store = MemoryGraphStore(db_path=Path(tmp_path / "memory_graph.db"))
    first = store.ingest_memory_entry(
        target="memory",
        action="add",
        content="Pytest is the main test runner.",
        session_id="s1",
    )
    second = store.ingest_memory_entry(
        target="memory",
        action="add",
        content="Pytest is the main test runner.",
        session_id="s1",
    )

    assert first == second
    rows = store._conn.execute("SELECT id, status FROM memory_nodes").fetchall()
    assert len(rows) == 1
    merge_events = store._conn.execute(
        "SELECT event_type, reason FROM memory_events WHERE event_type = 'merge'"
    ).fetchall()
    assert merge_events
    assert merge_events[0]["reason"] == "exact_duplicate"


def test_memory_graph_supersedes_on_replace(tmp_path):
    store = MemoryGraphStore(db_path=Path(tmp_path / "memory_graph.db"))
    old_id = store.ingest_memory_entry(
        target="memory",
        action="add",
        content="The test runner is nose.",
        session_id="s1",
    )
    new_id = store.ingest_memory_entry(
        target="memory",
        action="replace",
        old_text="nose",
        content="The test runner is pytest.",
        session_id="s1",
    )

    assert old_id != new_id
    old_row = store._conn.execute(
        "SELECT status, canonical_node_id FROM memory_nodes WHERE id = ?",
        (old_id,),
    ).fetchone()
    assert old_row["status"] == "superseded"
    assert old_row["canonical_node_id"] == new_id
    edge = store._conn.execute(
        "SELECT edge_type, src_id, dst_id FROM memory_edges WHERE src_id = ? AND dst_id = ?",
        (new_id, old_id),
    ).fetchone()
    assert edge["edge_type"] == "supersedes"


def test_memory_graph_archives_prior_session_summaries(tmp_path):
    store = MemoryGraphStore(db_path=Path(tmp_path / "memory_graph.db"))
    first = store.archive_session_summary(session_id="s1", summary_text="Old summary")
    second = store.archive_session_summary(session_id="s1", summary_text="New summary")

    assert first and second and first != second
    first_row = store._conn.execute(
        "SELECT status, archived_at FROM memory_nodes WHERE id = ?",
        (first,),
    ).fetchone()
    second_row = store._conn.execute(
        "SELECT status FROM memory_nodes WHERE id = ?",
        (second,),
    ).fetchone()
    assert first_row["status"] == "archived"
    assert first_row["archived_at"] is not None
    assert second_row["status"] == "active"


def test_memory_graph_prunes_low_value_archived_nodes(tmp_path):
    store = MemoryGraphStore(db_path=Path(tmp_path / "memory_graph.db"))
    node_id = store.add_node(
        node_type="observation",
        title="Low signal",
        content="Temporary note",
        status="archived",
        confidence=0.1,
        importance=0.1,
        session_id="s1",
        metadata={"target": "memory"},
    )

    pruned = store.prune_low_value_nodes(session_id="s1")

    assert node_id in pruned
    row = store._conn.execute(
        "SELECT status FROM memory_nodes WHERE id = ?",
        (node_id,),
    ).fetchone()
    assert row["status"] == "pruned"


def test_memory_graph_hybrid_retrieval_uses_semantic_signal(tmp_path):
    store = MemoryGraphStore(db_path=Path(tmp_path / "memory_graph.db"))

    vectors = {
        "Goal\nFix tokenizer bug in Hermes local MLX adapter": [1.0, 0.0, 0.0],
        "Observation\nTokenizer patch applied in transformers": [0.95, 0.05, 0.0],
        "Summary\nCompletely unrelated legal workspace note": [0.0, 1.0, 0.0],
        "fix tokenizer regex issue": [1.0, 0.0, 0.0],
    }

    def fake_embed(text: str):
        key = text.strip()
        return vectors.get(key, [0.0, 0.0, 1.0])

    store._embedder = fake_embed
    store._embedder_model = "test-embedder"

    store.add_node(
        node_type="goal",
        title="Goal",
        content="Fix tokenizer bug in Hermes local MLX adapter",
        session_id="s1",
        importance=0.9,
    )
    store.add_node(
        node_type="observation",
        title="Observation",
        content="Tokenizer patch applied in transformers",
        session_id="s1",
        importance=0.7,
    )
    store.add_node(
        node_type="summary",
        title="Summary",
        content="Completely unrelated legal workspace note",
        session_id="s1",
        importance=0.6,
    )

    rows = store.retrieve_frontier(query="fix tokenizer regex issue", session_id="s1", limit=3)

    assert rows
    assert rows[0]["type"] in {"goal", "observation"}
    assert "semantic" in rows[0]["retrieval_reasons"]
    assert rows[0]["semantic_score"] >= rows[-1]["semantic_score"]


def test_memory_graph_vec_index_tracks_embeddings_when_available(tmp_path):
    store = MemoryGraphStore(db_path=Path(tmp_path / "memory_graph.db"))
    if not store._vec_enabled:
        return

    store._embedder = lambda text: [1.0, 0.0, 0.0]
    store._embedder_model = "test-embedder"
    node_id = store.add_node(
        node_type="goal",
        title="Goal",
        content="Use sqlite vec for semantic retrieval",
        session_id="s1",
    )
    row = store._conn.execute(
        f"SELECT COUNT(*) AS c FROM {store._vec_table}"
    ).fetchone()
    assert row["c"] >= 1

    store._archive_node(node_id=node_id, reason="test")
    row = store._conn.execute(
        f"SELECT COUNT(*) AS c FROM {store._vec_table}"
    ).fetchone()
    assert row["c"] == 0


def test_memory_graph_builds_task_step_and_artifact_branch(tmp_path):
    store = MemoryGraphStore(db_path=Path(tmp_path / "memory_graph.db"))

    context = store.ensure_task_context(
        session_id="s1",
        task_id="t1",
        user_message="Inspect the workspace layout and identify the Obsidian vault.",
    )
    step = store.begin_tool_step(
        session_id="s1",
        task_id="t1",
        tool_name="terminal",
        tool_call_id="call_1",
        arguments={"command": "ls -la ~/Documents/Work/Matters"},
    )
    outcome = store.record_tool_outcome(
        session_id="s1",
        task_id="t1",
        step_id=step["step_id"],
        goal_id=step["goal_id"],
        task_root_id=step["task_root_id"],
        tool_name="terminal",
        tool_call_id="call_1",
        arguments={"command": "ls -la ~/Documents/Work/Matters"},
        result_text="Found .obsidian and several matter folders.",
        success=True,
        duration_seconds=0.15,
    )

    task_row = store._conn.execute("SELECT type FROM memory_nodes WHERE id = ?", (context["task_root_id"],)).fetchone()
    step_row = store._conn.execute("SELECT type FROM memory_nodes WHERE id = ?", (step["step_id"],)).fetchone()
    artifact_row = store._conn.execute("SELECT type FROM memory_nodes WHERE id = ?", (outcome["node_id"],)).fetchone()
    assert task_row["type"] == "task"
    assert step_row["type"] == "plan_step"
    assert artifact_row["type"] == "artifact"

    edges = store._conn.execute(
        """
        SELECT edge_type, src_id, dst_id
        FROM memory_edges
        WHERE (src_id = ? AND dst_id = ?)
           OR (src_id = ? AND dst_id = ?)
           OR (src_id = ? AND dst_id = ?)
        """,
        (
            context["task_root_id"], step["step_id"],
            step["step_id"], outcome["node_id"],
            outcome["node_id"], context["goal_id"],
        ),
    ).fetchall()
    edge_types = {row["edge_type"] for row in edges}
    assert {"depends_on", "produced", "supports"} <= edge_types


def test_memory_graph_resolves_blocker_and_prefers_active_branch(tmp_path):
    store = MemoryGraphStore(db_path=Path(tmp_path / "memory_graph.db"))
    store.ensure_task_context(
        session_id="s1",
        task_id="t1",
        user_message="Inspect the workspace layout and identify the Obsidian vault.",
    )

    failed_step = store.begin_tool_step(
        session_id="s1",
        task_id="t1",
        tool_name="terminal",
        tool_call_id="call_fail",
        arguments={"command": "ls missing-path"},
    )
    failed = store.record_tool_outcome(
        session_id="s1",
        task_id="t1",
        step_id=failed_step["step_id"],
        goal_id=failed_step["goal_id"],
        task_root_id=failed_step["task_root_id"],
        tool_name="terminal",
        tool_call_id="call_fail",
        arguments={"command": "ls missing-path"},
        result_text="Error: No such file or directory",
        success=False,
        duration_seconds=0.08,
    )

    success_step = store.begin_tool_step(
        session_id="s1",
        task_id="t1",
        tool_name="terminal",
        tool_call_id="call_ok",
        arguments={"command": "ls -la ~/Documents/Work/Matters"},
    )
    succeeded = store.record_tool_outcome(
        session_id="s1",
        task_id="t1",
        step_id=success_step["step_id"],
        goal_id=success_step["goal_id"],
        task_root_id=success_step["task_root_id"],
        tool_name="terminal",
        tool_call_id="call_ok",
        arguments={"command": "ls -la ~/Documents/Work/Matters"},
        result_text="Found the .obsidian vault in ~/Documents/Work/Matters.",
        success=True,
        duration_seconds=0.11,
    )

    blocker_row = store._conn.execute(
        "SELECT status, resolved_at FROM memory_nodes WHERE id = ?",
        (failed["node_id"],),
    ).fetchone()
    assert blocker_row["status"] == "resolved"
    assert blocker_row["resolved_at"] is not None

    resolve_edge = store._conn.execute(
        "SELECT edge_type FROM memory_edges WHERE src_id = ? AND dst_id = ?",
        (succeeded["node_id"], failed["node_id"]),
    ).fetchone()
    assert resolve_edge["edge_type"] == "resolves"

    frontier = store.retrieve_frontier(query="", session_id="s1", task_id="t1", limit=5)
    assert frontier
    assert any("branch" in row.get("retrieval_reasons", []) for row in frontier)
    assert frontier[0]["type"] in {"goal", "task", "artifact", "plan_step"}


def test_memory_graph_can_prefer_recent_temporal_retrieval(tmp_path):
    store = MemoryGraphStore(db_path=Path(tmp_path / "memory_graph.db"))
    recent_id = store.add_node(
        node_type="artifact",
        title="Recent task",
        content="Recent vault inspection task in the legal matters workspace",
        session_id="s1",
        importance=0.7,
    )
    old_id = store.add_node(
        node_type="artifact",
        title="Older task",
        content="Older vault inspection task from months ago",
        session_id="s1",
        importance=0.9,
    )
    store._conn.execute(
        "UPDATE memory_nodes SET updated_at = ?, created_at = ? WHERE id = ?",
        ("2026-03-08T12:00:00Z", "2026-03-08T12:00:00Z", recent_id),
    )
    store._conn.execute(
        "UPDATE memory_nodes SET updated_at = ?, created_at = ? WHERE id = ?",
        ("2025-01-01T12:00:00Z", "2025-01-01T12:00:00Z", old_id),
    )
    store._conn.commit()

    rows = store.retrieve_frontier(query="recent vault inspection", session_id="s1", limit=3)

    assert rows
    assert rows[0]["id"] == recent_id
    assert rows[0]["temporal_score"] >= rows[-1]["temporal_score"]


def test_memory_graph_can_target_older_archived_work_by_time_phrase(tmp_path):
    store = MemoryGraphStore(db_path=Path(tmp_path / "memory_graph.db"))
    recent_id = store.add_node(
        node_type="artifact",
        title="Current project",
        content="Current project work on Hermes memory graph",
        session_id="s1",
        importance=0.6,
    )
    archived_id = store.add_node(
        node_type="summary",
        title="Old project",
        content="Project from months ago involving the Obsidian vault migration",
        session_id="s1",
        status="archived",
        importance=0.8,
    )
    store._conn.execute(
        "UPDATE memory_nodes SET updated_at = ?, created_at = ? WHERE id = ?",
        ("2026-03-01T12:00:00Z", "2026-03-01T12:00:00Z", recent_id),
    )
    store._conn.execute(
        "UPDATE memory_nodes SET updated_at = ?, created_at = ?, archived_at = ? WHERE id = ?",
        ("2025-12-05T12:00:00Z", "2025-12-05T12:00:00Z", "2025-12-10T12:00:00Z", archived_id),
    )
    store._conn.commit()

    rows = store.retrieve_frontier(
        query="that one project we did 3 months ago vault migration",
        session_id="s1",
        limit=5,
    )

    assert rows
    assert rows[0]["id"] == archived_id
    assert rows[0]["status"] == "archived"


def test_memory_graph_maintenance_summarizes_and_prunes(tmp_path):
    store = MemoryGraphStore(db_path=Path(tmp_path / "memory_graph.db"))
    store.ensure_task_context(
        session_id="s1",
        task_id="t1",
        user_message="Finish the workspace migration and capture what changed.",
    )
    step = store.begin_tool_step(
        session_id="s1",
        task_id="t1",
        tool_name="terminal",
        tool_call_id="call_1",
        arguments={"command": "ls -la"},
    )
    artifact = store.record_tool_outcome(
        session_id="s1",
        task_id="t1",
        step_id=step["step_id"],
        goal_id=step["goal_id"],
        task_root_id=step["task_root_id"],
        tool_name="terminal",
        tool_call_id="call_1",
        arguments={"command": "ls -la"},
        result_text="Migration completed and files look correct.",
        success=True,
        duration_seconds=0.1,
    )["node_id"]
    blocker_step = store.begin_tool_step(
        session_id="s1",
        task_id="t1",
        tool_name="terminal",
        tool_call_id="call_2",
        arguments={"command": "test -d ~/.obsidian"},
    )
    blocker = store.record_tool_outcome(
        session_id="s1",
        task_id="t1",
        step_id=blocker_step["step_id"],
        goal_id=blocker_step["goal_id"],
        task_root_id=blocker_step["task_root_id"],
        tool_name="terminal",
        tool_call_id="call_2",
        arguments={"command": "test -d ~/.obsidian"},
        result_text="Error: path missing",
        success=False,
        duration_seconds=0.1,
    )["node_id"]
    resolved_step = store.begin_tool_step(
        session_id="s1",
        task_id="t1",
        tool_name="terminal",
        tool_call_id="call_3",
        arguments={"command": "ls -la ~/Documents/Work/Matters/.obsidian"},
    )
    store.record_tool_outcome(
        session_id="s1",
        task_id="t1",
        step_id=resolved_step["step_id"],
        goal_id=resolved_step["goal_id"],
        task_root_id=resolved_step["task_root_id"],
        tool_name="terminal",
        tool_call_id="call_3",
        arguments={"command": "ls -la ~/Documents/Work/Matters/.obsidian"},
        result_text="Found the vault and resolved the path issue.",
        success=True,
        duration_seconds=0.1,
    )
    low_value_id = store.add_node(
        node_type="artifact",
        title="Low signal artifact",
        content="Temporary output",
        session_id="s1",
        task_id="t1",
        confidence=0.2,
        importance=0.2,
        allow_duplicate=True,
    )

    old_ts = "2025-01-05T12:00:00Z"
    for node_id in [step["step_id"], artifact, blocker, blocker_step["step_id"], resolved_step["step_id"], low_value_id]:
        store._conn.execute(
            "UPDATE memory_nodes SET updated_at = ?, created_at = ?, resolved_at = COALESCE(resolved_at, ?) WHERE id = ?",
            (old_ts, old_ts, old_ts, node_id),
        )
    store._conn.commit()

    result = store.run_maintenance(session_id="s1", task_id="t1", stale_days=30, prune_max=10)

    assert result["summary_id"] is not None
    summary_row = store._conn.execute(
        "SELECT type FROM memory_nodes WHERE id = ?",
        (result["summary_id"],),
    ).fetchone()
    assert summary_row["type"] == "summary"

    archived_row = store._conn.execute(
        "SELECT status FROM memory_nodes WHERE id = ?",
        (artifact,),
    ).fetchone()
    assert archived_row["status"] == "archived"

    low_value_row = store._conn.execute(
        "SELECT status FROM memory_nodes WHERE id = ?",
        (low_value_id,),
    ).fetchone()
    assert low_value_row["status"] in {"archived", "pruned"}


def test_memory_graph_semantic_maintenance_auto_merges_low_risk_types(tmp_path):
    store = MemoryGraphStore(db_path=Path(tmp_path / "memory_graph.db"))
    vectors = {
        "Workspace note\nObsidian vault is in ~/Documents/Work/Matters": [1.0, 0.0, 0.0],
        "Workspace note\nFound the .obsidian vault under ~/Documents/Work/Matters": [0.99, 0.01, 0.0],
    }

    def fake_embed(text: str):
        return vectors[text.strip()]

    store._embedder = fake_embed
    store._embedder_model = "test-embedder"

    first = store.add_node(
        node_type="observation",
        title="Workspace note",
        content="Obsidian vault is in ~/Documents/Work/Matters",
        session_id="s1",
        allow_duplicate=True,
    )
    second = store.add_node(
        node_type="observation",
        title="Workspace note",
        content="Found the .obsidian vault under ~/Documents/Work/Matters",
        session_id="s1",
        allow_duplicate=True,
    )

    result = store.run_maintenance(session_id="s1", stale_days=3650, prune_max=0)

    assert result["semantic_merge"]["auto_merged"]
    merged_row = store._conn.execute(
        "SELECT status, canonical_node_id FROM memory_nodes WHERE id = ?",
        (first,),
    ).fetchone()
    other_row = store._conn.execute(
        "SELECT status, canonical_node_id FROM memory_nodes WHERE id = ?",
        (second,),
    ).fetchone()
    assert {merged_row["status"], other_row["status"]} == {"active", "superseded"}
    assert first == other_row["canonical_node_id"] or second == merged_row["canonical_node_id"]


def test_memory_graph_semantic_maintenance_marks_merge_candidates_for_protected_types(tmp_path):
    store = MemoryGraphStore(db_path=Path(tmp_path / "memory_graph.db"))
    vectors = {
        "Goal\nInspect the workspace and confirm the vault path": [1.0, 0.0, 0.0],
        "Goal\nVerify where the Obsidian vault lives in the workspace": [0.985, 0.015, 0.0],
    }

    def fake_embed(text: str):
        return vectors[text.strip()]

    store._embedder = fake_embed
    store._embedder_model = "test-embedder"

    first = store.add_node(
        node_type="goal",
        title="Goal",
        content="Inspect the workspace and confirm the vault path",
        session_id="s1",
        allow_duplicate=True,
    )
    second = store.add_node(
        node_type="goal",
        title="Goal",
        content="Verify where the Obsidian vault lives in the workspace",
        session_id="s1",
        allow_duplicate=True,
    )

    result = store.run_maintenance(session_id="s1", stale_days=3650, prune_max=0)

    assert not result["semantic_merge"]["auto_merged"]
    edge = store._conn.execute(
        "SELECT edge_type FROM memory_edges WHERE (src_id = ? AND dst_id = ?) OR (src_id = ? AND dst_id = ?)",
        (first, second, second, first),
    ).fetchone()
    assert edge["edge_type"] == "merge_candidate"
    rows = store._conn.execute(
        "SELECT status FROM memory_nodes WHERE id IN (?, ?)",
        (first, second),
    ).fetchall()
    assert {row["status"] for row in rows} == {"active"}


def test_memory_graph_synthesizes_hypothesis_from_disparate_memories(tmp_path):
    store = MemoryGraphStore(db_path=Path(tmp_path / "memory_graph.db"))
    vectors = {
        "Observation\nVault migration touched the Matters workspace paths": [1.0, 0.0, 0.0],
        "Artifact\nObsidian lookup failed because workspace paths changed": [0.72, 0.28, 0.0],
    }

    def fake_embed(text: str):
        return vectors[text.strip()]

    store._embedder = fake_embed
    store._embedder_model = "test-embedder"

    first = store.add_node(
        node_type="observation",
        title="Observation",
        content="Vault migration touched the Matters workspace paths",
        session_id="s1",
        task_id="t1",
        allow_duplicate=True,
    )
    second = store.add_node(
        node_type="artifact",
        title="Artifact",
        content="Obsidian lookup failed because workspace paths changed",
        session_id="s1",
        task_id="t2",
        allow_duplicate=True,
    )

    created = store.synthesize_hypotheses(session_id="s1")

    assert created
    hypothesis = store._conn.execute(
        "SELECT type, metadata_json FROM memory_nodes WHERE id = ?",
        (created[0],),
    ).fetchone()
    assert hypothesis["type"] == "hypothesis"
    metadata = store._parse_metadata(hypothesis["metadata_json"])
    assert set(metadata["source_ids"]) == {first, second}


def test_memory_graph_safe_test_can_resolve_hypothesis(tmp_path):
    store = MemoryGraphStore(db_path=Path(tmp_path / "memory_graph.db"))
    hypothesis_id = store.add_node(
        node_type="hypothesis",
        title="Synthesized hypothesis",
        content="The vault path issue may be tied to Matters workspace migration.",
        session_id="s1",
        task_id="t1",
        metadata={
            "source_ids": ["a", "b"],
            "predicted_terms": ["vault", "matters"],
            "test_strategy": "safe_read_only_validation",
        },
        allow_duplicate=True,
    )

    created = store.evaluate_hypotheses_from_tool(
        session_id="s1",
        task_id="t1",
        tool_name="terminal",
        result_text="ls output confirms the vault exists under the Matters workspace.",
        safe_test=True,
    )

    assert created
    hypothesis_row = store._conn.execute(
        "SELECT status, resolved_at FROM memory_nodes WHERE id = ?",
        (hypothesis_id,),
    ).fetchone()
    assert hypothesis_row["status"] == "resolved"
    assert hypothesis_row["resolved_at"] is not None
    edge = store._conn.execute(
        "SELECT edge_type FROM memory_edges WHERE src_id = ? AND dst_id = ?",
        (created[0], hypothesis_id),
    ).fetchone()
    assert edge["edge_type"] == "supports"


def test_memory_graph_working_memory_packet_and_expansion(tmp_path):
    store = MemoryGraphStore(db_path=Path(tmp_path / "memory_graph.db"))
    goal_id = store.record_session_goal(session_id="s1", task_id="t1", content="Inspect the vault path")
    step = store.begin_tool_step(
        session_id="s1",
        task_id="t1",
        tool_name="terminal",
        tool_call_id="call_1",
        arguments={"command": "ls -la ~/Documents/Work/Matters"},
    )
    artifact_id = store.record_tool_outcome(
        session_id="s1",
        task_id="t1",
        step_id=step["step_id"],
        goal_id=goal_id,
        task_root_id=step["task_root_id"],
        tool_name="terminal",
        tool_call_id="call_1",
        arguments={"command": "ls -la ~/Documents/Work/Matters"},
        result_text="Found the .obsidian vault under ~/Documents/Work/Matters.",
        success=True,
        duration_seconds=0.1,
    )["node_id"]

    packet = store.retrieve_working_memory(query="vault path", session_id="s1", task_id="t1", limit=4)
    assert packet
    assert all("node_id" in item for item in packet)

    expanded = store.expand_memory_node(artifact_id)
    assert expanded is not None
    assert expanded["node"]["id"] == artifact_id
    assert "neighbors" in expanded
    assert "events" in expanded


def test_memory_graph_review_actions_apply_prune_and_merge(tmp_path):
    store = MemoryGraphStore(db_path=Path(tmp_path / "memory_graph.db"))

    target_id = store.add_node(
        node_type="artifact",
        title="Stale scratch note",
        content="Temporary note from dream review.",
        session_id="s1",
        importance=0.2,
        confidence=0.2,
    )
    prune_candidate_id = store.add_node(
        node_type="decision",
        title="Dream prune candidate",
        content="Archive stale scratch note",
        session_id="s1",
        source_kind="dreamcycle_prune_candidate",
        metadata={"target_node_id": target_id},
        allow_duplicate=True,
    )

    canonical_id = store.add_node(
        node_type="goal",
        title="Canonical goal",
        content="Find the Obsidian vault in the Matters workspace.",
        session_id="s1",
    )
    duplicate_id = store.add_node(
        node_type="goal",
        title="Duplicate goal",
        content="Locate the Obsidian vault in the Matters workspace.",
        session_id="s1",
        allow_duplicate=True,
    )

    assert store.review_prune_candidate(candidate_id=prune_candidate_id, action="approve", session_id="s1")
    assert store.review_merge_candidate(
        canonical_node_id=canonical_id,
        duplicate_node_id=duplicate_id,
        action="approve",
        session_id="s1",
    )

    target_row = store._conn.execute(
        "SELECT status FROM memory_nodes WHERE id = ?",
        (target_id,),
    ).fetchone()
    prune_row = store._conn.execute(
        "SELECT status FROM memory_nodes WHERE id = ?",
        (prune_candidate_id,),
    ).fetchone()
    duplicate_row = store._conn.execute(
        "SELECT status, canonical_node_id FROM memory_nodes WHERE id = ?",
        (duplicate_id,),
    ).fetchone()

    assert target_row["status"] == "archived"
    assert prune_row["status"] == "resolved"
    assert duplicate_row["status"] == "superseded"
    assert duplicate_row["canonical_node_id"] == canonical_id
