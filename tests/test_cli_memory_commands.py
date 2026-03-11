from types import SimpleNamespace

from cli import HermesCLI


class _FakeGraph:
    def __init__(self):
        self.expand_calls = []
        self.prune_reviews = []
        self.merge_reviews = []

    def retrieve_working_memory(self, **kwargs):
        return [
            {
                "node_id": "node-1",
                "type": "goal",
                "status": "active",
                "title": "Investigate vault location",
                "brief": "User expects the Obsidian vault under ~/Documents/Work/Matters.",
                "why_relevant": "active goal",
                "score": 0.91,
            }
        ]

    def expand_memory_node(self, node_id):
        self.expand_calls.append(node_id)
        return {
            "node": {
                "id": node_id,
                "type": "goal",
                "status": "active",
                "title": "Investigate vault location",
                "content": "Look for .obsidian in the Matters workspace.",
            },
            "neighbors": [
                {
                    "edge_type": "supports",
                    "title": "Vault under Matters",
                    "related_id": "node-2",
                }
            ],
            "events": [
                {
                    "event_type": "retrieve",
                    "timestamp": "2026-03-10T12:00:00Z",
                    "reason": "manual expand",
                }
            ],
        }

    def list_nodes(self, **kwargs):
        source_kind = kwargs.get("source_kind")
        node_type = kwargs.get("node_type")
        if source_kind == "dreamcycle_prune_candidate":
            return [
                {
                    "id": "prune-1",
                    "content": "Archive stale scratch note",
                    "metadata_json": '{"target_node_id":"node-stale"}',
                }
            ]
        if node_type == "hypothesis":
            return [
                {
                    "id": "hyp-1",
                    "status": "active",
                    "title": "Hypothesis",
                    "content": "Workspace likely contains an Obsidian vault.",
                }
            ]
        return [
            {
                "id": "dream-1",
                "type": "hypothesis",
                "status": "active",
                "title": "Dream hypothesis",
                "content": "Test whether related memories indicate a hidden vault path.",
                "source_kind": "dreamcycle_hypothesis",
            }
        ]

    def list_events(self, **kwargs):
        return [
            {
                "timestamp": "2026-03-10T12:01:00Z",
                "node_id": "node-a",
                "related_node_id": "node-b",
                "payload_json": '{"similarity":0.97}',
            }
        ]

    def _parse_metadata(self, raw):
        import json

        return json.loads(raw) if raw else {}

    def memory_dashboard(self, **kwargs):
        return {
            "total_nodes": 12,
            "active_goals": 1,
            "active_blockers": 0,
            "active_hypotheses": 2,
            "dream_artifacts": 3,
            "prune_candidates": 1,
            "merge_candidates": 1,
        }

    def review_prune_candidate(self, **kwargs):
        self.prune_reviews.append(kwargs)
        return True

    def review_merge_candidate(self, **kwargs):
        self.merge_reviews.append(kwargs)
        return True


class _FakeAgent:
    def __init__(self):
        self.session_id = "sess-1"
        self._memory_graph = _FakeGraph()
        self.dream_calls = []

    def run_dream_cycle(self, **kwargs):
        self.dream_calls.append(kwargs)
        return {
            "packet_size": 2,
            "heuristic_hypotheses": [{"id": "h1"}],
            "dream_hypotheses": [],
            "experiment_plans": [{"id": "p1"}],
            "prune_candidates": [{"id": "pr1"}],
        }


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.agent = _FakeAgent()
    cli_obj.session_id = "sess-1"
    cli_obj.conversation_history = []
    cli_obj._dream_idle_enabled = False
    cli_obj._dream_idle_seconds = 1
    cli_obj._dream_idle_cadence = 2
    cli_obj._last_activity_monotonic = 0.0
    cli_obj._last_dream_monotonic = 0.0
    cli_obj._agent_running = False
    cli_obj._should_exit = False
    cli_obj._dream_review_items = []
    return cli_obj


def test_dream_status_and_auto_toggle(capsys):
    cli_obj = _make_cli()

    cli_obj.process_command("/dream status")
    cli_obj.process_command("/dream auto on")
    cli_obj.process_command("/dream auto off")

    output = capsys.readouterr().out
    assert "DreamCycle status" in output
    assert "Idle DreamCycle enabled" in output
    assert "Idle DreamCycle disabled" in output
    assert cli_obj._dream_idle_enabled is False


def test_memory_status_summary_includes_review_counts():
    cli_obj = _make_cli()
    cli_obj._dream_idle_enabled = True

    summary = cli_obj._get_memory_status_summary()

    assert "memory goals:1" in summary
    assert "blockers:0" in summary
    assert "hyp:2" in summary
    assert "review:2" in summary
    assert "dream:auto" in summary


def test_memory_dispatch_and_review_commands(capsys):
    cli_obj = _make_cli()

    cli_obj.process_command("/memory dashboard")
    cli_obj.process_command("/memory packet vault")
    cli_obj.process_command("/memory expand node-1")
    cli_obj.process_command("/memory hypotheses")
    cli_obj.process_command("/memory merge-candidates")
    cli_obj.process_command("/memory merge-approve node-a node-b")
    cli_obj.process_command("/memory merge-reject node-a node-b")
    cli_obj.process_command("/memory prune-candidates")
    cli_obj.process_command("/memory prune-approve prune-1")
    cli_obj.process_command("/memory prune-reject prune-1")
    cli_obj.process_command("/memory dreams")
    cli_obj.process_command("/dream review")
    cli_obj.process_command("/dream review expand 1")
    cli_obj.process_command("/dream review approve 1")
    cli_obj.process_command("/dream review reject 2")

    output = capsys.readouterr().out
    assert "Memory dashboard" in output
    assert "Working memory packet" in output
    assert "Investigate vault location" in output
    assert "node-a -> node-b" in output
    assert "prune-1" in output
    assert "Recent dream artifacts" in output
    assert "Dream review queue" in output
    assert "actions=expand|approve|reject" in output
    assert "Prune candidate approved" in output
    assert "Merge candidate rejected" in output
    assert cli_obj.agent._memory_graph.expand_calls == ["node-1", "node-stale"]
    assert len(cli_obj.agent._memory_graph.prune_reviews) == 3
    assert len(cli_obj.agent._memory_graph.merge_reviews) == 3


def test_idle_dreamcycle_runs_when_thresholds_met():
    cli_obj = _make_cli()
    cli_obj._dream_idle_enabled = True
    cli_obj._last_activity_monotonic = 0.0
    cli_obj._last_dream_monotonic = 0.0

    import cli as cli_module

    original_time = cli_module.time.monotonic
    try:
        cli_module.time.monotonic = lambda: 10.0
        cli_obj._maybe_run_idle_dreamcycle()
    finally:
        cli_module.time.monotonic = original_time

    assert cli_obj.agent.dream_calls == [{"focus": "idle background maintenance", "max_hypotheses": 2}]
    assert cli_obj._last_dream_monotonic == 10.0
