"""hermes-graph — SQLite-backed structured memory graph plugin.

A MemoryProvider that wraps MemoryGraphStore with dreamcycle synthesis and
subconscious selection.  Stores observations, hypotheses, goals, decisions,
and session summaries in a local SQLite graph with FTS5, optional vector
search (via local MLX embeddings), and graph-aware retrieval fusion.

Config in $HERMES_HOME/config.yaml (profile-scoped):
  plugins:
    hermes-graph:
      db_path: $HERMES_HOME/memory_graph.db
      dreamcycle_on_idle: true
      subconscious_expand: 2
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from .memory_graph import MemoryGraphStore
from .dreamcycle import DreamCycle
from .subconscious import SubconsciousSelector

logger = logging.getLogger(__name__)


def _hermes_home() -> Path:
    return Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))


class HermesGraphProvider(MemoryProvider):
    """MemoryProvider backed by local SQLite memory graph."""

    def __init__(self) -> None:
        self._store: Optional[MemoryGraphStore] = None
        self._session_id: str = ""
        self._hermes_home: Path = _hermes_home()
        self._db_path: Optional[Path] = None
        self._dreamcycle_enabled: bool = True
        self._subconscious_expand: int = 2
        self._sync_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "hermes-graph"

    def is_available(self) -> bool:
        # Always available — SQLite is built-in, no external service needed
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._hermes_home = Path(kwargs.get("hermes_home", _hermes_home()))

        # Load plugin-specific config
        config = self._load_config()
        db_path_str = config.get("db_path", "")
        if db_path_str:
            self._db_path = Path(db_path_str)
        else:
            self._db_path = self._hermes_home / "memory_graph.db"
        self._dreamcycle_enabled = config.get("dreamcycle_on_idle", True)
        self._subconscious_expand = int(config.get("subconscious_expand", 2))

        self._store = MemoryGraphStore(db_path=self._db_path)
        logger.info(
            "HermesGraphProvider initialized: db=%s session=%s",
            self._db_path, session_id,
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "memory_graph_retrieve",
                "description": (
                    "Retrieve relevant memories from the structured memory graph. "
                    "Uses hybrid FTS + semantic + graph-aware fusion to find "
                    "observations, hypotheses, goals, decisions, and session summaries."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query describing what to recall",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum results to return (default 6)",
                            "default": 6,
                        },
                        "prefer_recent": {
                            "type": "boolean",
                            "description": "Boost recently created or accessed memories",
                            "default": False,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "memory_graph_store",
                "description": (
                    "Store a new memory node in the graph. Use for important "
                    "observations, decisions, hypotheses, or goals that should "
                    "persist across sessions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The memory content to store",
                        },
                        "title": {
                            "type": "string",
                            "description": "Short title for the memory",
                        },
                        "node_type": {
                            "type": "string",
                            "enum": [
                                "observation", "hypothesis", "goal",
                                "decision", "blocker", "artifact", "summary",
                            ],
                            "description": "Category of memory",
                            "default": "observation",
                        },
                        "importance": {
                            "type": "number",
                            "description": "0.0-1.0 importance score",
                            "default": 0.5,
                        },
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "memory_graph_expand",
                "description": (
                    "Expand a specific memory node to see its full content, "
                    "neighboring nodes, and event history."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_id": {
                            "type": "string",
                            "description": "The memory node ID to expand",
                        },
                    },
                    "required": ["node_id"],
                },
            },
            {
                "name": "memory_graph_dashboard",
                "description": (
                    "Show memory graph statistics: node counts by type/status, "
                    "recent activity, and health metrics."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "memory_graph_dreamcycle",
                "description": (
                    "Run a dreamcycle synthesis pass: generate hypotheses, "
                    "experiment plans, and prune candidates from existing memories. "
                    "Use during idle time or when asked to reflect."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "focus": {
                            "type": "string",
                            "description": "Optional focus topic for synthesis",
                            "default": "",
                        },
                    },
                },
            },
            {
                "name": "memory_graph_maintenance",
                "description": (
                    "Run memory maintenance: archive session summaries, "
                    "synthesize hypotheses, merge semantic duplicates, and "
                    "prune low-value nodes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary_text": {
                            "type": "string",
                            "description": "Session summary text to archive",
                            "default": "",
                        },
                    },
                },
            },
        ]

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "db_path",
                "description": "Path to SQLite database (default: $HERMES_HOME/memory_graph.db)",
                "default": "",
            },
            {
                "key": "dreamcycle_on_idle",
                "description": "Run dreamcycle synthesis during idle periods",
                "default": True,
                "choices": [True, False],
            },
            {
                "key": "subconscious_expand",
                "description": "Number of nodes to expand via subconscious selector (0 to disable)",
                "default": 2,
            },
        ]

    def save_config(self, values: dict, hermes_home: str) -> None:
        config_path = Path(hermes_home) / "hermes-graph.json"
        config_path.write_text(json.dumps(values, indent=2))

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._store is None:
            return json.dumps({"error": "HermesGraphProvider not initialized"})

        try:
            if tool_name == "memory_graph_retrieve":
                return self._handle_retrieve(args)
            elif tool_name == "memory_graph_store":
                return self._handle_store(args)
            elif tool_name == "memory_graph_expand":
                return self._handle_expand(args)
            elif tool_name == "memory_graph_dashboard":
                return self._handle_dashboard()
            elif tool_name == "memory_graph_dreamcycle":
                return self._handle_dreamcycle(args)
            elif tool_name == "memory_graph_maintenance":
                return self._handle_maintenance(args)
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as exc:
            logger.exception("HermesGraph tool %s failed", tool_name)
            return json.dumps({"error": str(exc)})

    def shutdown(self) -> None:
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._store = None

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def system_prompt_block(self) -> str:
        if self._store is None:
            return ""
        try:
            stats = self._store.memory_dashboard()
            total = stats.get("total_nodes", 0)
            active = stats.get("active_nodes", 0)
            return (
                "# Hermes Memory Graph\n"
                f"Local SQLite memory graph active — {active} active nodes, "
                f"{total} total. Use memory_graph_* tools to retrieve, store, "
                "expand, run dreamcycle synthesis, or view dashboard."
            )
        except Exception:
            return "# Hermes Memory Graph\nLocal memory graph active."

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._store or not query.strip():
            return ""
        try:
            packet = self._store.retrieve_working_memory(
                query=query,
                session_id=session_id or self._session_id,
                limit=4,
            )
            if not packet:
                return ""

            # Subconscious expansion
            if self._subconscious_expand > 0 and len(packet) > 1:
                selector = SubconsciousSelector()
                expand_ids = selector.choose_expansions(
                    query=query, packet=packet, max_expand=self._subconscious_expand,
                )
                expanded_details = []
                for nid in expand_ids:
                    detail = self._store.expand_memory_node(nid)
                    if detail and detail.get("node"):
                        content = detail["node"].get("content", "")
                        if len(content) > 300:
                            content = content[:300] + "..."
                        expanded_details.append(f"  [{nid}]: {content}")

                if expanded_details:
                    expanded_block = "\n".join(expanded_details)
                else:
                    expanded_block = ""
            else:
                expanded_block = ""

            lines = ["## Memory Graph Context"]
            for item in packet:
                lines.append(
                    f"- **{item['title']}** ({item['type']}/{item['status']}) "
                    f"[{item['node_id']}]: {item['brief']}"
                )
            if expanded_block:
                lines.append("\n### Expanded Nodes")
                lines.append(expanded_block)
            return "\n".join(lines)
        except Exception as exc:
            logger.debug("HermesGraph prefetch failed: %s", exc)
            return ""

    def sync_turn(
        self, user_content: str, assistant_content: str, *, session_id: str = ""
    ) -> None:
        """Non-blocking: ingest the turn's key observations in a background thread."""
        if not self._store:
            return

        store = self._store
        sid = session_id or self._session_id

        def _sync():
            try:
                # Record user message as a lightweight observation
                if user_content and len(user_content.strip()) > 20:
                    store.add_node(
                        node_type="observation",
                        title="User turn",
                        content=user_content[:500],
                        confidence=0.6,
                        importance=0.3,
                        session_id=sid,
                        source_kind="turn_sync",
                    )
            except Exception as exc:
                logger.debug("HermesGraph sync_turn failed: %s", exc)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=3.0)
        self._sync_thread = threading.Thread(target=_sync, daemon=True)
        self._sync_thread.start()

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._store:
            return
        try:
            # Run maintenance on session end
            self._store.run_maintenance(
                session_id=self._session_id,
                summary_text="",
            )
        except Exception as exc:
            logger.debug("HermesGraph on_session_end maintenance failed: %s", exc)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract key insights before context compression discards messages."""
        if not self._store:
            return ""
        try:
            # Record that a compression happened
            self._store.record_event(
                event_type="context_compress",
                session_id=self._session_id,
                reason="pre_compress_hook",
                payload={"message_count": len(messages)},
            )
            return ""
        except Exception:
            return ""

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes into the graph."""
        if not self._store or not content:
            return
        try:
            self._store.ingest_memory_entry(
                target=target,
                content=content,
                action=action,
                session_id=self._session_id,
                source_ref="builtin_memory_write",
            )
        except Exception as exc:
            logger.debug("HermesGraph on_memory_write failed: %s", exc)

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    def _handle_retrieve(self, args: dict) -> str:
        query = args.get("query", "")
        limit = int(args.get("limit", 6))
        prefer_recent = bool(args.get("prefer_recent", False))

        packet = self._store.retrieve_working_memory(
            query=query,
            session_id=self._session_id,
            limit=limit,
        )

        # Subconscious expansion on retrieval too
        expanded = {}
        if self._subconscious_expand > 0 and packet:
            selector = SubconsciousSelector()
            expand_ids = selector.choose_expansions(
                query=query, packet=packet, max_expand=self._subconscious_expand,
            )
            for nid in expand_ids:
                detail = self._store.expand_memory_node(nid)
                if detail:
                    expanded[nid] = detail

        return json.dumps({
            "memories": packet,
            "expanded": {
                k: {
                    "node": v.get("node", {}),
                    "neighbor_count": len(v.get("neighbors", [])),
                }
                for k, v in expanded.items()
            },
            "total_results": len(packet),
        }, ensure_ascii=False, default=str)

    def _handle_store(self, args: dict) -> str:
        content = args.get("content", "")
        if not content.strip():
            return json.dumps({"error": "content is required"})

        node_id = self._store.add_node(
            node_type=args.get("node_type", "observation"),
            content=content,
            title=args.get("title", ""),
            confidence=0.8,
            importance=float(args.get("importance", 0.5)),
            session_id=self._session_id,
            source_kind="tool_store",
        )
        return json.dumps({"node_id": node_id, "status": "stored"})

    def _handle_expand(self, args: dict) -> str:
        node_id = args.get("node_id", "")
        if not node_id:
            return json.dumps({"error": "node_id is required"})

        detail = self._store.expand_memory_node(node_id)
        if not detail:
            return json.dumps({"error": f"Node {node_id} not found"})

        return json.dumps(detail, ensure_ascii=False, default=str)

    def _handle_dashboard(self) -> str:
        stats = self._store.memory_dashboard()
        return json.dumps(stats, ensure_ascii=False, default=str)

    def _handle_dreamcycle(self, args: dict) -> str:
        focus = args.get("focus", "")
        dc = DreamCycle(self._store, session_id=self._session_id)
        result = dc.run_once(focus=focus)
        return json.dumps(result, ensure_ascii=False, default=str)

    def _handle_maintenance(self, args: dict) -> str:
        summary_text = args.get("summary_text", "")
        result = self._store.run_maintenance(
            session_id=self._session_id,
            summary_text=summary_text,
        )
        return json.dumps(result, ensure_ascii=False, default=str)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_config(self) -> Dict[str, Any]:
        """Load plugin config from hermes-graph.json or config.yaml."""
        # Try dedicated config file first
        config_file = self._hermes_home / "hermes-graph.json"
        if config_file.exists():
            try:
                return json.loads(config_file.read_text())
            except Exception:
                pass

        # Fall back to config.yaml plugins section
        config_yaml = self._hermes_home / "config.yaml"
        if config_yaml.exists():
            try:
                import yaml
                with open(config_yaml) as f:
                    full = yaml.safe_load(f) or {}
                return full.get("plugins", {}).get("hermes-graph", {})
            except Exception:
                pass

        return {}


def register(ctx) -> None:
    """Called by the memory plugin discovery system."""
    ctx.register_memory_provider(HermesGraphProvider())
