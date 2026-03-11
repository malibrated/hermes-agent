---
sidebar_position: 5
title: "Memory Graph Spec"
description: "Implementation specification for Hermes' SQLite-backed structured memory graph and consolidation pipeline."
---

# Memory Graph Spec

This document specifies a local-first, SQLite-backed structured memory system for Hermes.

The design goal is to improve:

- continuation robustness on long or interrupted tasks
- consolidation of noisy transcript history into compact task state
- retrieval of unresolved goals, evidence, decisions, and artifacts
- auditability of merge, summarize, archive, and prune decisions

This is an adjunct memory layer. It does not replace Hermes' transcript history, existing file-backed memories, or session search in v1.

---

## 1. Design Goals

- Preserve high-signal task state across long tool loops, interruptions, session resets, and context compression.
- Move from transcript accumulation to structured consolidation.
- Record not only what memory changed, but why, and in what active task context.
- Keep the system local-first, easy to inspect, and operationally lightweight.
- Support later code-structure linking without requiring AST-driven storage in v1.

## 2. Non-Goals

- No graph database in v1.
- No replacement of existing `~/.hermes/memories/` markdown memory files.
- No full AST or symbol graph integration in v1.
- No aggressive hard deletion during normal agent execution.

---

## 3. Architectural Overview

The memory graph has four layers:

1. Conversation Layer
   Hermes' existing session transcript remains the raw event stream.

2. Structured Memory Layer
   Nodes and edges represent goals, evidence, decisions, artifacts, and summaries.

3. Consolidation Layer
   Deterministic policies merge, summarize, archive, and rarely prune nodes.

4. Retrieval Layer
   Bounded retrieval returns the active frontier relevant to the current task.

The consolidation layer is the core of the system. Graph structure alone is not enough; every merge/archive/prune decision must be auditable.

---

## 4. Storage Choice

Use SQLite as the source of truth.

Recommended settings:

- WAL mode enabled
- foreign keys enabled
- schema version tracking
- JSON metadata stored as text JSON
- background maintenance only
- no hard deletion during active sessions

Recommended path:

```text
~/.hermes/memory_graph.db
```

Why SQLite:

- zero-infra local-first deployment
- transactional updates for consolidation operations
- FTS5 support for cheap text retrieval
- recursive CTE support for lineage and branch reconstruction
- easy inspection and backup

---

## 5. Core Data Model

### 5.1 Nodes

A node is a memory unit representing a durable or semi-durable piece of agent knowledge.

Initial node types:

- `goal`
- `plan_step`
- `artifact`
- `observation`
- `hypothesis`
- `decision`
- `summary`
- `blocker`
- `result`

### 5.2 Edges

An edge is a typed relationship between nodes.

Initial edge types:

- `decomposes_into`
- `depends_on`
- `supports`
- `contradicts`
- `produced`
- `tests`
- `resolves`
- `supersedes`
- `summarizes`
- `related_to`

### 5.3 Events

An event is an append-only record describing a memory operation or state change.

Initial event types:

- `create`
- `merge`
- `summarize`
- `supersede`
- `archive`
- `prune`
- `split`
- `confidence_update`
- `status_update`
- `retrieve`

### 5.4 Node Status

Each node has a lifecycle status:

- `active`
- `resolved`
- `rejected`
- `superseded`
- `archived`
- `pruned`

`archived` means hidden from default retrieval but retained.

`pruned` means removed from normal retrieval due to low value, but still traceable via event history.

---

## 6. Timestamp Requirements

Timestamps are mandatory on nodes and events.

Node fields:

- `created_at`
- `updated_at`
- `last_accessed_at`
- `last_supported_at`
- `last_contradicted_at`
- `resolved_at`
- `archived_at`
- `expires_at`

Edge fields:

- `created_at`
- `updated_at`

Event fields:

- `timestamp`

These timestamps support:

- recency-aware retrieval
- decay and archival policies
- timeline reconstruction
- consolidation explainability

---

## 7. SQLite Schema

### 7.1 `memory_nodes`

```sql
CREATE TABLE memory_nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    status TEXT NOT NULL,
    title TEXT,
    content TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    importance REAL NOT NULL DEFAULT 0.5,
    canonical_node_id TEXT,
    summary_level INTEGER NOT NULL DEFAULT 0,
    session_id TEXT,
    task_id TEXT,
    source_kind TEXT,
    source_ref TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed_at TEXT,
    last_supported_at TEXT,
    last_contradicted_at TEXT,
    resolved_at TEXT,
    archived_at TEXT,
    expires_at TEXT,
    FOREIGN KEY (canonical_node_id) REFERENCES memory_nodes(id)
);
```

Recommended indexes:

```sql
CREATE INDEX idx_memory_nodes_type_status ON memory_nodes(type, status);
CREATE INDEX idx_memory_nodes_task_status ON memory_nodes(task_id, status);
CREATE INDEX idx_memory_nodes_session_created ON memory_nodes(session_id, created_at);
CREATE INDEX idx_memory_nodes_canonical ON memory_nodes(canonical_node_id);
CREATE INDEX idx_memory_nodes_accessed ON memory_nodes(last_accessed_at);
CREATE INDEX idx_memory_nodes_rank ON memory_nodes(importance, confidence);
```

### 7.2 `memory_edges`

```sql
CREATE TABLE memory_edges (
    id TEXT PRIMARY KEY,
    src_id TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (src_id) REFERENCES memory_nodes(id),
    FOREIGN KEY (dst_id) REFERENCES memory_nodes(id)
);
```

Recommended indexes:

```sql
CREATE INDEX idx_memory_edges_src_type ON memory_edges(src_id, edge_type);
CREATE INDEX idx_memory_edges_dst_type ON memory_edges(dst_id, edge_type);
CREATE INDEX idx_memory_edges_type ON memory_edges(edge_type);
```

### 7.3 `memory_events`

```sql
CREATE TABLE memory_events (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    trigger TEXT NOT NULL,
    policy_name TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    explanation TEXT NOT NULL,
    reversible INTEGER NOT NULL DEFAULT 1,
    active_goal_ids_json TEXT NOT NULL DEFAULT '[]',
    context_json TEXT NOT NULL DEFAULT '{}',
    timestamp TEXT NOT NULL
);
```

Recommended indexes:

```sql
CREATE INDEX idx_memory_events_type_time ON memory_events(event_type, timestamp);
CREATE INDEX idx_memory_events_trigger_time ON memory_events(trigger, timestamp);
```

### 7.4 `memory_event_nodes`

```sql
CREATE TABLE memory_event_nodes (
    event_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    role TEXT NOT NULL,
    PRIMARY KEY (event_id, node_id, role),
    FOREIGN KEY (event_id) REFERENCES memory_events(id),
    FOREIGN KEY (node_id) REFERENCES memory_nodes(id)
);
```

Valid `role` values:

- `input`
- `output`
- `affected`
- `evidence`

Recommended indexes:

```sql
CREATE INDEX idx_memory_event_nodes_event_role ON memory_event_nodes(event_id, role);
CREATE INDEX idx_memory_event_nodes_node_role ON memory_event_nodes(node_id, role);
```

### 7.5 `memory_node_fts`

```sql
CREATE VIRTUAL TABLE memory_node_fts USING fts5(
    node_id UNINDEXED,
    title,
    content
);
```

This is optional for v1 runtime retrieval, but recommended for developer inspection and future hybrid retrieval.

---

## 8. Event Semantics

The event log is not optional metadata. It is part of the system of record.

Every non-trivial consolidation action must emit an event containing:

- what changed
- why it changed
- under what trigger
- under what active goal context
- which nodes were involved
- whether the action is reversible
- what policy made the decision

Required fields:

- `event_type`
- `trigger`
- `policy_name`
- `reason_code`
- `explanation`
- `active_goal_ids_json`
- `context_json`
- `timestamp`

`context_json` should include enough context to explain the decision later, for example:

- current `task_id`
- current `session_id`
- model/provider
- active frontier node ids
- recent message span or turn ids if available
- compression pressure or budget signal if relevant

Example reason codes:

- `duplicate_fact`
- `same_goal_reintroduced`
- `branch_resolved_summarized`
- `contradicted_by_new_evidence`
- `inactive_after_task_completion`
- `low_confidence_low_importance_leaf`
- `replaced_by_canonical_summary`

---

## 9. Write Triggers

The system should be event-driven, not naively continuous.

### Immediate capture triggers

- end of assistant turn
- completion of a tool batch
- explicit decision or blocker detection
- major error or recovery branch
- before transcript compression
- session end, reset, or interruption

### Deferred consolidation triggers

- every 3-5 turns
- after long tool loops
- on task completion
- when compression triggers
- during idle/background maintenance

Hot path writes should stay lightweight:

- create candidate nodes
- attach provenance
- perform cheap exact dedupe only if obvious

Heavier merge/summarize/archive decisions should be deferred.

---

## 10. Candidate Extraction

Extract structured candidates from:

- user requests
- assistant commitments
- tool calls and tool outputs
- file or artifact discoveries
- commands and command results
- failures, stack traces, and error messages
- decisions and conclusions
- unresolved blockers

v1 extraction should be rule-based and conservative.

Examples:

- user asks for investigation -> `goal`
- assistant says it will inspect/test/fix -> `plan_step`
- tool output reveals a file path -> `artifact`
- test failure or command error -> `observation` or `result`
- assistant states likely cause -> `hypothesis`
- assistant chooses a fix -> `decision`

---

## 11. Consolidation Policy

### 11.1 Guiding Rules

- merge aggressively when identity is clear
- summarize regularly for resolved branches
- archive often
- hard-prune rarely

### 11.2 Operations

#### Dedupe

Use when two nodes are materially the same fact, goal, or artifact.

Result:

- preserve one canonical node
- attach provenance from both
- emit `merge` or `confidence_update` event

#### Merge

Use when nodes refer to the same underlying entity but contain additive detail.

Result:

- update canonical node or emit merged replacement
- preserve ancestry

#### Summarize

Use when a branch is resolved or a set of low-level nodes can be compacted safely.

Result:

- create `summary` node
- create `summarizes` edges
- archive source branch or mark superseded where appropriate

#### Supersede

Use when new evidence refines or invalidates earlier memory.

Result:

- old node becomes `superseded`
- new node links via `supersedes`

#### Archive

Use when a node is not needed in active retrieval but may remain useful later.

#### Prune

Use rarely, only when the node is:

- low confidence
- low importance
- weakly connected
- not active
- not pinned
- not recently retrieved
- already summarized or redundant

Prune should usually mean "leave lineage, remove from standard retrieval", not "hard delete".

---

## 12. Merge/Archive/Prune Explainability

Every consolidation action must record:

- input node ids
- output node ids
- reason code
- explanation
- active goals at the time
- policy name
- reversibility
- evidence references

This allows answering:

- why was this node archived?
- why were these two nodes merged?
- what summary replaced this branch?
- under what task context was this treated as low-value?

Opaque pruning is not acceptable.

---

## 13. Retrieval Model

Retrieval should be bounded and deterministic in v1.

Inputs:

- current user message
- current `task_id`
- current `session_id`
- active node ids if known

Output:

- active goals
- unresolved blockers
- latest decisions
- recent evidence supporting the current branch
- summary nodes for older resolved branches
- key artifacts

Retrieval ranking factors:

- node status
- task/session match
- recency
- importance
- confidence
- reuse frequency
- light centrality or edge connectivity

Default budget:

- 8-20 nodes
- plus 1-3 summary nodes

Archived nodes are excluded by default unless explicitly requested.

---

## 14. Integration With Existing Hermes Systems

This memory graph should be integrated gradually.

Do not replace these systems in v1:

- transcript/session history
- existing markdown memories
- session search

Use the memory graph first to improve:

- continuation robustness
- unresolved-task persistence
- context compression quality
- branch summarization

Only later should it become a direct input to the main prompt builder.

---

## 15. Continuation Support

The initial practical use case is preventing the agent from stopping too early.

The structured memory graph should provide:

- active `goal` nodes
- unresolved `plan_step` nodes
- unresolved `blocker` nodes
- `decision` nodes indicating chosen branch
- `result` nodes indicating completed evidence collection

At end of turn, Hermes can ask:

- does the active frontier still show unresolved work?
- did the assistant produce only an acknowledgement or stub?
- was tool work completed but not summarized to the user?

This complements the current text heuristics in `run_agent.py` and gives a more reliable continuation signal over time.

---

## 16. Compression Support

Before transcript compression:

- extract structured candidates from the soon-to-be-compressed message region
- consolidate them
- create summary nodes where appropriate

This prevents context compression from discarding latent task structure.

Summary nodes should preserve ancestry to the underlying branch.

---

## 17. Reversibility and Repair

The system must support repair of bad consolidation decisions.

Requirements:

- inspect node ancestry
- inspect event history
- reverse archival when needed
- split improperly merged branches
- reconstruct branch history from events

This is one reason hard deletion should not occur during active workflows.

---

## 18. Operational Safety

Recommended invariants:

- no hard deletion during active sessions
- pruning only in deferred/background maintenance
- every consolidation action is logged
- foreign key integrity enabled
- schema versioning required
- periodic integrity checks

Maintenance tasks:

- archive stale low-signal leaves
- compact fully summarized branches
- rebuild FTS if required
- optional `VACUUM` in maintenance only

---

## 19. Suggested Python Service API

Suggested core service:

```python
class MemoryGraphStore:
    def create_node(...)
    def create_edge(...)
    def record_event(...)
    def extract_candidates_from_turn(...)
    def consolidate(...)
    def retrieve_frontier(...)
    def inspect_node(...)
    def inspect_event(...)
    def reconstruct_lineage(...)
```

Suggested additional helper layers:

- `MemoryGraphExtractor`
- `MemoryGraphConsolidator`
- `MemoryGraphRetriever`
- `MemoryGraphInspector`

This keeps ingestion, consolidation, retrieval, and inspection decoupled.

---

## 20. Rollout Plan

### Phase 1

- add SQLite schema
- add storage layer
- add event recording
- add inspection/debug utilities

### Phase 2

- add rule-based candidate extraction
- add node/edge creation after turns and tool batches
- add exact dedupe and conservative merge/archive

### Phase 3

- add retrieval API
- integrate retrieval with continuation and compression

### Phase 4

- add richer merge/supersede logic
- add better ranking
- optionally add artifact/code symbol links later

---

## 21. First Implementation Priorities

Build in this order:

1. SQLite schema and access layer
2. event log and inspection tooling
3. candidate extraction from end-of-turn and post-tool hooks
4. conservative consolidation and archival
5. active frontier retrieval
6. continuation integration
7. compression integration

This ordering gives immediate value on long-running tasks without forcing a full prompt-system rewrite.

---

## 22. Success Criteria

The memory graph is successful if it improves:

- continuation without needing "please continue"
- preservation of unresolved state after compression
- reduced repeated investigation work
- better recall of prior conclusions in long sessions
- auditability of merge/archive/prune decisions

It should do so without:

- noticeable hot-path latency
- opaque loss of distinctions
- runaway node growth

---

## 23. Future Extensions

Once the base graph is stable, future work can add:

- symbol-level code nodes derived from AST or tree-sitter
- richer artifact linking
- hybrid retrieval combining FTS and structural retrieval
- learned or LLM-assisted consolidation ranking
- developer-facing inspection UI

Those should remain extensions to the core structured memory model, not prerequisites for v1.
