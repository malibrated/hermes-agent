"""SQLite-backed structured memory graph sidecar for Hermes."""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
import uuid
import math
import calendar
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes")) / "memory_graph.db"
SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memory_graph_schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_graph_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_nodes (
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
    embedding_json TEXT,
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

CREATE TABLE IF NOT EXISTS memory_edges (
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

CREATE TABLE IF NOT EXISTS memory_events (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    node_id TEXT,
    related_node_id TEXT,
    session_id TEXT,
    task_id TEXT,
    reason TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    timestamp TEXT NOT NULL,
    FOREIGN KEY (node_id) REFERENCES memory_nodes(id),
    FOREIGN KEY (related_node_id) REFERENCES memory_nodes(id)
);

CREATE INDEX IF NOT EXISTS idx_memory_nodes_type_status ON memory_nodes(type, status);
CREATE INDEX IF NOT EXISTS idx_memory_nodes_task_status ON memory_nodes(task_id, status);
CREATE INDEX IF NOT EXISTS idx_memory_nodes_session_created ON memory_nodes(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_memory_nodes_rank ON memory_nodes(importance, confidence);
CREATE INDEX IF NOT EXISTS idx_memory_edges_src_type ON memory_edges(src_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_memory_edges_dst_type ON memory_edges(dst_id, edge_type);
CREATE INDEX IF NOT EXISTS idx_memory_events_node_time ON memory_events(node_id, timestamp);
"""

FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS memory_nodes_fts USING fts5(
    title,
    content,
    content=memory_nodes,
    content_rowid=rowid
);

CREATE TRIGGER IF NOT EXISTS memory_nodes_fts_insert AFTER INSERT ON memory_nodes BEGIN
    INSERT INTO memory_nodes_fts(rowid, title, content)
    VALUES (new.rowid, new.title, new.content);
END;

CREATE TRIGGER IF NOT EXISTS memory_nodes_fts_delete AFTER DELETE ON memory_nodes BEGIN
    INSERT INTO memory_nodes_fts(memory_nodes_fts, rowid, title, content)
    VALUES('delete', old.rowid, old.title, old.content);
END;

CREATE TRIGGER IF NOT EXISTS memory_nodes_fts_update AFTER UPDATE ON memory_nodes BEGIN
    INSERT INTO memory_nodes_fts(memory_nodes_fts, rowid, title, content)
    VALUES('delete', old.rowid, old.title, old.content);
    INSERT INTO memory_nodes_fts(rowid, title, content)
    VALUES (new.rowid, new.title, new.content);
END;
"""


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _normalize_text(value: str) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _truncate_text(value: str, limit: int = 1200) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 16].rstrip() + "\n...[truncated]"


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


def _significant_tokens(text: str) -> List[str]:
    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "into", "under", "over",
        "there", "here", "they", "them", "their", "about", "using", "uses", "used",
        "project", "task", "work", "workspace", "file", "path", "found", "find",
        "agent", "memory", "result", "output", "note", "summary", "observation",
    }
    tokens = []
    seen = set()
    for token in re.findall(r"[A-Za-z0-9_./~-]+", text or ""):
        tok = token.lower()
        if len(tok) < 3 or tok in stopwords or tok.isdigit():
            continue
        if tok in seen:
            continue
        seen.add(tok)
        tokens.append(tok)
    return tokens


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _parse_utc_timestamp(value: Optional[str]) -> float:
    if not value:
        return 0.0


def _utc_now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _dt_to_utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _node_time(row: Dict[str, Any]) -> float:
    return (
        _parse_utc_timestamp(row.get("last_accessed_at"))
        or _parse_utc_timestamp(row.get("updated_at"))
        or _parse_utc_timestamp(row.get("created_at"))
    )


def _extract_temporal_preferences(query: str) -> Dict[str, Any]:
    text = (query or "").strip()
    lowered = text.lower()
    prefs: Dict[str, Any] = {
        "clean_query": text,
        "time_from": None,
        "time_to": None,
        "prefer_recent": False,
        "prefer_archived": False,
        "prefer_older": False,
    }
    if not lowered:
        return prefs

    now = _utc_now_dt()
    patterns = [
        (r"\brecent\b", {"prefer_recent": True, "lookback_days": 14}),
        (r"\blately\b", {"prefer_recent": True, "lookback_days": 14}),
        (r"\brecently\b", {"prefer_recent": True, "lookback_days": 14}),
        (r"\bcurrent\b", {"prefer_recent": True, "lookback_days": 7}),
        (r"\blast\s+week\b", {"prefer_recent": True, "lookback_days": 7}),
        (r"\blast\s+month\b", {"prefer_recent": True, "lookback_days": 31}),
        (r"\byesterday\b", {"prefer_recent": True, "lookback_days": 2}),
        (r"\btoday\b", {"prefer_recent": True, "lookback_days": 1}),
        (r"\bolder\b", {"prefer_older": True, "prefer_archived": True}),
        (r"\bold\b", {"prefer_older": True, "prefer_archived": True}),
        (r"\bmonths?\s+ago\b", {"prefer_older": True, "prefer_archived": True}),
    ]
    clean = text
    for pattern, effect in patterns:
        if re.search(pattern, lowered):
            clean = re.sub(pattern, " ", clean, flags=re.IGNORECASE)
            for key, value in effect.items():
                prefs[key] = value

    ago_match = re.search(r"\b(\d+)\s+(day|week|month|year)s?\s+ago\b", lowered)
    if ago_match:
        amount = int(ago_match.group(1))
        unit = ago_match.group(2)
        if unit == "day":
            delta = timedelta(days=amount)
            span = timedelta(days=max(2, amount))
        elif unit == "week":
            delta = timedelta(weeks=amount)
            span = timedelta(days=10)
        elif unit == "month":
            delta = timedelta(days=30 * amount)
            span = timedelta(days=21)
        else:
            delta = timedelta(days=365 * amount)
            span = timedelta(days=45)
        center = now - delta
        prefs["time_from"] = _dt_to_utc_text(center - span)
        prefs["time_to"] = _dt_to_utc_text(center + span)
        prefs["prefer_older"] = True
        prefs["prefer_archived"] = True
        clean = re.sub(r"\b\d+\s+(day|week|month|year)s?\s+ago\b", " ", clean, flags=re.IGNORECASE)

    if prefs.get("lookback_days"):
        prefs["time_from"] = prefs["time_from"] or _dt_to_utc_text(now - timedelta(days=int(prefs["lookback_days"])))
        prefs["time_to"] = prefs["time_to"] or _dt_to_utc_text(now + timedelta(days=1))

    stopwords = {
        "that", "one", "we", "did", "the", "a", "an", "our", "this", "those",
        "these", "project", "task", "work", "from", "about", "around",
    }
    filtered_tokens = [tok for tok in re.findall(r"[A-Za-z0-9_/-]+", clean) if tok.lower() not in stopwords]
    clean = " ".join(filtered_tokens) if filtered_tokens else re.sub(r"\s+", " ", clean).strip(" ,.")
    prefs["clean_query"] = clean
    return prefs
    try:
        return calendar.timegm(time.strptime(value, "%Y-%m-%dT%H:%M:%SZ"))
    except Exception:
        return 0.0


class MemoryGraphStore:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._vec_enabled = False
        self._vec_error: Optional[str] = None
        self._vec_table = "memory_nodes_vec"
        self._init_schema()
        self._init_vec_extension()

        self._embedder = None
        try:
            from agent.local_mlx import local_mlx_embed, local_mlx_configured
            if local_mlx_configured() or os.getenv("LOCAL_MLX_EMBED_MODEL"):
                self._embedder = local_mlx_embed
        except Exception as exc:
            logger.debug("Memory graph embedder unavailable: %s", exc)
        self._embedder_model = os.getenv("LOCAL_MLX_EMBED_MODEL", "mlx-community/nomicai-modernbert-embed-base-4bit")

    def _edge_relevance_weight(self, edge_type: str) -> float:
        weights = {
            "supports": 1.0,
            "depends_on": 0.95,
            "produced": 0.85,
            "resolves": 0.85,
            "blocks": 0.9,
            "summarizes": 0.65,
            "supersedes": 0.55,
            "derived_from": 0.6,
        }
        return weights.get(edge_type, 0.4)

    def _init_schema(self) -> None:
        cursor = self._conn.cursor()
        cursor.executescript(SCHEMA_SQL)
        row = cursor.execute("SELECT version FROM memory_graph_schema_version LIMIT 1").fetchone()
        if row is None:
            cursor.execute("INSERT INTO memory_graph_schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        try:
            cursor.execute("SELECT * FROM memory_nodes_fts LIMIT 0")
        except sqlite3.OperationalError:
            cursor.executescript(FTS_SQL)
        self._conn.commit()

    def _init_vec_extension(self) -> None:
        try:
            import sqlite_vec
        except Exception as exc:
            self._vec_error = str(exc)
            logger.debug("sqlite-vec unavailable: %s", exc)
            return
        try:
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._vec_enabled = True
        except Exception as exc:
            self._vec_error = str(exc)
            logger.debug("sqlite-vec load failed: %s", exc)

    def _setting(self, key: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT value FROM memory_graph_settings WHERE key = ?",
            (key,),
        ).fetchone()
        return str(row["value"]) if row else None

    def _set_setting(self, key: str, value: str) -> None:
        self._conn.execute(
            """
            INSERT INTO memory_graph_settings(key, value)
            VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )

    def _maybe_embed(self, text: str) -> Optional[str]:
        if not self._embedder or not text.strip():
            return None
        try:
            return json.dumps(self._embedder(text))
        except Exception as exc:
            logger.debug("Memory graph embedding failed: %s", exc)
            return None

    def _embed_metadata(self) -> Dict[str, Any]:
        return {
            "embedding_model": self._embedder_model,
            "embedding_backend": "mlx-embeddings",
            "embedding_created_at": _utc_now(),
        }

    def _decode_embedding(self, value: Any) -> Optional[List[float]]:
        if not value:
            return None
        try:
            data = json.loads(value) if isinstance(value, str) else value
        except Exception:
            return None
        if not isinstance(data, list):
            return None
        try:
            return [float(v) for v in data]
        except Exception:
            return None

    def _normalize_embedding(self, values: List[float]) -> List[float]:
        norm = math.sqrt(sum(v * v for v in values))
        if norm <= 0:
            return values
        return [v / norm for v in values]

    def _embedding_compatible(self, metadata: Dict[str, Any], vector: Optional[List[float]], query_dim: int) -> bool:
        if not vector or len(vector) != query_dim:
            return False
        model = (metadata.get("embedding_model") or "").strip()
        return not model or model == self._embedder_model

    def _ensure_vec_table(self, dim: int) -> bool:
        if not self._vec_enabled or dim <= 0:
            return False
        current_dim = self._setting("vec_dim")
        if current_dim and int(current_dim) != dim:
            self._conn.execute(f"DROP TABLE IF EXISTS {self._vec_table}")
            self._set_setting("vec_dim", str(dim))
            self._set_setting("vec_model", self._embedder_model)
            self._conn.commit()
        exists = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
            (self._vec_table,),
        ).fetchone()
        if not exists:
            self._conn.execute(
                f"CREATE VIRTUAL TABLE {self._vec_table} USING vec0(embedding float[{dim}])"
            )
            self._set_setting("vec_dim", str(dim))
            self._set_setting("vec_model", self._embedder_model)
            self._conn.commit()
            self._rebuild_vec_index()
        return True

    def _rebuild_vec_index(self) -> None:
        if not self._vec_enabled:
            return
        dim = self._setting("vec_dim")
        if not dim:
            return
        dim_int = int(dim)
        self._conn.execute(f"DELETE FROM {self._vec_table}")
        rows = self._conn.execute(
            """
            SELECT rowid, embedding_json, metadata_json
            FROM memory_nodes
            WHERE embedding_json IS NOT NULL
            """
        ).fetchall()
        for row in rows:
            metadata = self._parse_metadata(row["metadata_json"])
            vector = self._decode_embedding(row["embedding_json"])
            if not self._embedding_compatible(metadata, vector, dim_int):
                continue
            self._conn.execute(
                f"INSERT OR REPLACE INTO {self._vec_table}(rowid, embedding) VALUES (?, ?)",
                (int(row["rowid"]), json.dumps(vector)),
            )
        self._conn.commit()

    def _upsert_vec_row(self, *, node_id: str, vector: Optional[List[float]]) -> None:
        if not self._vec_enabled or not vector:
            return
        if not self._ensure_vec_table(len(vector)):
            return
        row = self._conn.execute(
            "SELECT rowid FROM memory_nodes WHERE id = ?",
            (node_id,),
        ).fetchone()
        if not row:
            return
        try:
            self._conn.execute(f"DELETE FROM {self._vec_table} WHERE rowid = ?", (int(row["rowid"]),))
        except Exception:
            pass
        self._conn.execute(
            f"INSERT INTO {self._vec_table}(rowid, embedding) VALUES (?, ?)",
            (int(row["rowid"]), json.dumps(vector)),
        )

    def _delete_vec_row(self, *, node_id: str) -> None:
        if not self._vec_enabled:
            return
        row = self._conn.execute(
            "SELECT rowid FROM memory_nodes WHERE id = ?",
            (node_id,),
        ).fetchone()
        if not row:
            return
        try:
            self._conn.execute(f"DELETE FROM {self._vec_table} WHERE rowid = ?", (int(row["rowid"]),))
        except Exception:
            pass

    def _fts_candidates(
        self,
        *,
        query: str,
        session_id: Optional[str],
        task_id: Optional[str],
        time_from: Optional[str],
        time_to: Optional[str],
        include_archived: bool,
        limit: int,
    ) -> List[Dict[str, Any]]:
        if not query.strip():
            return []
        params: List[Any] = [query.strip()]
        sql = """
            SELECT n.*, bm25(memory_nodes_fts) AS rank_score
            FROM memory_nodes_fts
            JOIN memory_nodes n ON n.rowid = memory_nodes_fts.rowid
            WHERE memory_nodes_fts MATCH ?
              AND n.status IN ('active', 'resolved'{archived})
        """
        sql = sql.format(archived=", 'archived'" if include_archived else "")
        if session_id:
            sql += " AND (n.session_id = ? OR n.session_id IS NULL)"
            params.append(session_id)
        if task_id:
            sql += " AND (n.task_id = ? OR n.task_id IS NULL)"
            params.append(task_id)
        if time_from:
            sql += " AND COALESCE(n.updated_at, n.created_at) >= ?"
            params.append(time_from)
        if time_to:
            sql += " AND COALESCE(n.updated_at, n.created_at) <= ?"
            params.append(time_to)
        sql += " ORDER BY rank_score, n.importance DESC, n.updated_at DESC LIMIT ?"
        params.append(limit)
        return [dict(row) for row in self._conn.execute(sql, tuple(params)).fetchall()]

    def _semantic_candidates(
        self,
        *,
        query: str,
        session_id: Optional[str],
        task_id: Optional[str],
        time_from: Optional[str],
        time_to: Optional[str],
        include_archived: bool,
        limit: int,
    ) -> List[Dict[str, Any]]:
        if not self._embedder or not query.strip():
            return []
        try:
            query_vec = self._normalize_embedding(self._embedder(query.strip()))
        except Exception as exc:
            logger.debug("Memory graph query embedding failed: %s", exc)
            return []
        if self._vec_enabled and self._ensure_vec_table(len(query_vec)):
            try:
                params: List[Any] = [json.dumps(query_vec), max(limit * 4, 32)]
                sql = f"""
                    SELECT n.*, v.distance
                    FROM {self._vec_table} v
                    JOIN memory_nodes n ON n.rowid = v.rowid
                    WHERE v.embedding MATCH ?
                      AND k = ?
                      AND n.status IN ('active', 'resolved'{archived})
                """
                sql = sql.format(archived=", 'archived'" if include_archived else "")
                if session_id:
                    sql += " AND (n.session_id = ? OR n.session_id IS NULL)"
                    params.append(session_id)
                if task_id:
                    sql += " AND (n.task_id = ? OR n.task_id IS NULL)"
                    params.append(task_id)
                if time_from:
                    sql += " AND COALESCE(n.updated_at, n.created_at) >= ?"
                    params.append(time_from)
                if time_to:
                    sql += " AND COALESCE(n.updated_at, n.created_at) <= ?"
                    params.append(time_to)
                sql += " ORDER BY v.distance ASC"
                rows = [dict(row) for row in self._conn.execute(sql, tuple(params)).fetchall()]
                for row in rows:
                    distance = float(row.get("distance", 1.0) or 1.0)
                    row["semantic_raw_score"] = max(0.0, 1.0 - distance)
                rows.sort(key=lambda row: row.get("semantic_raw_score", 0.0), reverse=True)
                return rows[:limit]
            except Exception as exc:
                logger.debug("sqlite-vec semantic retrieval failed, falling back to brute force: %s", exc)
        params: List[Any] = []
        sql = """
            SELECT *
            FROM memory_nodes
            WHERE status IN ('active', 'resolved'{archived})
              AND embedding_json IS NOT NULL
        """
        sql = sql.format(archived=", 'archived'" if include_archived else "")
        if session_id:
            sql += " AND (session_id = ? OR session_id IS NULL)"
            params.append(session_id)
        if task_id:
            sql += " AND (task_id = ? OR task_id IS NULL)"
            params.append(task_id)
        if time_from:
            sql += " AND COALESCE(updated_at, created_at) >= ?"
            params.append(time_from)
        if time_to:
            sql += " AND COALESCE(updated_at, created_at) <= ?"
            params.append(time_to)
        rows = self._conn.execute(sql, tuple(params)).fetchall()
        scored: List[Dict[str, Any]] = []
        for row in rows:
            row_dict = dict(row)
            metadata = self._parse_metadata(row_dict.get("metadata_json"))
            vector = self._decode_embedding(row_dict.get("embedding_json"))
            if not self._embedding_compatible(metadata, vector, len(query_vec)):
                continue
            score = sum(a * b for a, b in zip(query_vec, vector))
            if score <= 0:
                continue
            row_dict["semantic_raw_score"] = float(score)
            scored.append(row_dict)
        scored.sort(key=lambda row: row.get("semantic_raw_score", 0.0), reverse=True)
        return scored[:limit]

    def _fallback_text_candidates(
        self,
        *,
        query: str,
        session_id: Optional[str],
        task_id: Optional[str],
        time_from: Optional[str],
        time_to: Optional[str],
        include_archived: bool,
        limit: int,
    ) -> List[Dict[str, Any]]:
        tokens = [tok.lower() for tok in re.findall(r"[A-Za-z0-9_/-]+", query or "") if len(tok) >= 3]
        if not tokens:
            return []
        params: List[Any] = []
        sql = """
            SELECT *
            FROM memory_nodes
            WHERE status IN ('active', 'resolved'{archived})
        """.format(archived=", 'archived'" if include_archived else "")
        if session_id:
            sql += " AND (session_id = ? OR session_id IS NULL)"
            params.append(session_id)
        if task_id:
            sql += " AND (task_id = ? OR task_id IS NULL)"
            params.append(task_id)
        if time_from:
            sql += " AND COALESCE(updated_at, created_at) >= ?"
            params.append(time_from)
        if time_to:
            sql += " AND COALESCE(updated_at, created_at) <= ?"
            params.append(time_to)
        rows = [dict(row) for row in self._conn.execute(sql, tuple(params)).fetchall()]
        scored: List[Dict[str, Any]] = []
        for row in rows:
            hay = _normalize_text(f"{row.get('title') or ''} {row.get('content') or ''}")
            hits = sum(1 for tok in tokens if tok in hay)
            if hits <= 0:
                continue
            row["rank_score"] = float(len(tokens) - hits)
            row["fallback_text_score"] = hits / max(1, len(tokens))
            scored.append(row)
        scored.sort(
            key=lambda row: (
                float(row.get("fallback_text_score", 0.0) or 0.0),
                float(row.get("importance", 0.0) or 0.0),
                row.get("updated_at") or "",
            ),
            reverse=True,
        )
        return scored[:limit]

    def _recency_score(self, row: Dict[str, Any]) -> float:
        ts = (
            _parse_utc_timestamp(row.get("last_accessed_at"))
            or _parse_utc_timestamp(row.get("updated_at"))
            or _parse_utc_timestamp(row.get("created_at"))
        )
        if not ts:
            return 0.0
        age_days = max(0.0, (time.time() - ts) / 86400.0)
        return 1.0 / (1.0 + age_days / 7.0)

    def _status_score(self, status: str) -> float:
        if status == "active":
            return 1.0
        if status == "resolved":
            return 0.7
        if status == "archived":
            return 0.55
        return 0.0

    def _normalize_ranked_scores(self, rows: List[Dict[str, Any]], key: str, reverse: bool = True) -> Dict[str, float]:
        if not rows:
            return {}
        values = [float(row.get(key, 0.0) or 0.0) for row in rows]
        low = min(values)
        high = max(values)
        scores: Dict[str, float] = {}
        for row in rows:
            val = float(row.get(key, 0.0) or 0.0)
            if high == low:
                norm = 1.0
            else:
                raw = (val - low) / (high - low)
                norm = raw if reverse else 1.0 - raw
            scores[str(row["id"])] = _clamp(norm)
        return scores

    def _graph_score(self, node_id: str) -> float:
        row = self._conn.execute(
            """
            SELECT COUNT(*) AS edge_count
            FROM memory_edges
            WHERE src_id = ? OR dst_id = ?
            """,
            (node_id, node_id),
        ).fetchone()
        count = int(row["edge_count"] or 0) if row else 0
        return _clamp(count / 5.0)

    def _active_branch_seed_ids(
        self,
        *,
        session_id: Optional[str],
        task_id: Optional[str],
        time_from: Optional[str],
        time_to: Optional[str],
        include_archived: bool,
        limit: int = 8,
    ) -> List[str]:
        params: List[Any] = []
        sql = """
            SELECT id
            FROM memory_nodes
            WHERE status IN ('active', 'resolved'{archived})
              AND type IN ('goal', 'task', 'blocker', 'plan_step', 'decision', 'artifact', 'result')
        """
        sql = sql.format(archived=", 'archived'" if include_archived else "")
        if session_id:
            sql += " AND (session_id = ? OR session_id IS NULL)"
            params.append(session_id)
        if task_id:
            sql += " AND (task_id = ? OR task_id IS NULL)"
            params.append(task_id)
        if time_from:
            sql += " AND COALESCE(updated_at, created_at) >= ?"
            params.append(time_from)
        if time_to:
            sql += " AND COALESCE(updated_at, created_at) <= ?"
            params.append(time_to)
        sql += """
            ORDER BY
              CASE type
                WHEN 'goal' THEN 0
                WHEN 'task' THEN 1
                WHEN 'blocker' THEN 2
                WHEN 'decision' THEN 3
                WHEN 'plan_step' THEN 4
                WHEN 'artifact' THEN 5
                WHEN 'result' THEN 6
                ELSE 7
              END,
              importance DESC,
              updated_at DESC
            LIMIT ?
        """
        params.append(limit)
        return [str(row["id"]) for row in self._conn.execute(sql, tuple(params)).fetchall()]

    def _graph_neighbor_candidates(
        self,
        *,
        seed_ids: List[str],
        session_id: Optional[str],
        task_id: Optional[str],
        time_from: Optional[str],
        time_to: Optional[str],
        include_archived: bool,
        limit: int,
    ) -> List[Dict[str, Any]]:
        if not seed_ids:
            return []
        placeholders = ",".join("?" for _ in seed_ids)
        params: List[Any] = []
        sql = f"""
            SELECT
                n.*,
                e.edge_type,
                e.weight AS edge_weight
            FROM memory_edges e
            JOIN memory_nodes n
              ON n.id = CASE
                WHEN e.src_id IN ({placeholders}) THEN e.dst_id
                ELSE e.src_id
              END
            WHERE (e.src_id IN ({placeholders}) OR e.dst_id IN ({placeholders}))
              AND n.status IN ('active', 'resolved'{{archived}})
        """
        sql = sql.format(archived=", 'archived'" if include_archived else "")
        params.extend(seed_ids)
        params.extend(seed_ids)
        params.extend(seed_ids)
        if session_id:
            sql += " AND (n.session_id = ? OR n.session_id IS NULL)"
            params.append(session_id)
        if task_id:
            sql += " AND (n.task_id = ? OR n.task_id IS NULL)"
            params.append(task_id)
        if time_from:
            sql += " AND COALESCE(n.updated_at, n.created_at) >= ?"
            params.append(time_from)
        if time_to:
            sql += " AND COALESCE(n.updated_at, n.created_at) <= ?"
            params.append(time_to)
        sql += " ORDER BY n.importance DESC, n.updated_at DESC"
        rows = [dict(row) for row in self._conn.execute(sql, tuple(params)).fetchall()]

        scored: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            node_id = str(row["id"])
            raw = _clamp(float(row.get("edge_weight") or 1.0) * self._edge_relevance_weight(str(row.get("edge_type") or "")))
            current = scored.get(node_id)
            if current is None or raw > float(current.get("graph_raw_score", 0.0) or 0.0):
                enriched = dict(row)
                enriched["graph_raw_score"] = raw
                scored[node_id] = enriched

        # Keep seeds in the graph candidate set too so active goals/blocks can win even without strong text match.
        if seed_ids:
            placeholders = ",".join("?" for _ in seed_ids)
            archived_fragment = ", 'archived'" if include_archived else ""
            seed_rows = self._conn.execute(
                f"SELECT * FROM memory_nodes WHERE id IN ({placeholders}) AND status IN ('active', 'resolved'{archived_fragment})",
                tuple(seed_ids),
            ).fetchall()
            for row in seed_rows:
                node_id = str(row["id"])
                enriched = dict(row)
                enriched["graph_raw_score"] = max(
                    float(scored.get(node_id, {}).get("graph_raw_score", 0.0) or 0.0),
                    1.0,
                )
                scored[node_id] = enriched

        out = list(scored.values())
        out.sort(
            key=lambda row: (
                float(row.get("graph_raw_score", 0.0) or 0.0),
                float(row.get("importance", 0.0) or 0.0),
                row.get("updated_at") or "",
            ),
            reverse=True,
        )
        return out[:limit]

    def _fuse_candidates(
        self,
        *,
        lexical: List[Dict[str, Any]],
        semantic: List[Dict[str, Any]],
        graph: List[Dict[str, Any]],
        session_id: Optional[str],
        task_id: Optional[str],
        prefer_recent: bool,
        prefer_older: bool,
        include_archived: bool,
        limit: int,
    ) -> List[Dict[str, Any]]:
        candidates: Dict[str, Dict[str, Any]] = {}
        for row in lexical + semantic + graph:
            candidates.setdefault(str(row["id"]), dict(row))
        fts_scores = self._normalize_ranked_scores(lexical, "rank_score", reverse=False)
        semantic_scores = self._normalize_ranked_scores(semantic, "semantic_raw_score", reverse=True)
        graph_scores = self._normalize_ranked_scores(graph, "graph_raw_score", reverse=True)
        ranked: List[Dict[str, Any]] = []
        for node_id, row in candidates.items():
            importance_score = _clamp(float(row.get("importance") or 0.0))
            confidence_score = _clamp(float(row.get("confidence") or 0.0))
            recency_score = self._recency_score(row)
            status_score = self._status_score(str(row.get("status") or ""))
            graph_degree_score = self._graph_score(node_id)
            session_match = 1.0 if session_id and row.get("session_id") == session_id else 0.0
            task_match = 1.0 if task_id and row.get("task_id") == task_id else 0.0
            temporal_score = recency_score if prefer_recent else (1.0 - recency_score if prefer_older else 0.5)
            archived_bonus = 0.08 if include_archived and str(row.get("status") or "") == "archived" else 0.0
            type_name = str(row.get("type") or "")
            type_bias = {
                "goal": 0.12,
                "task": 0.10,
                "blocker": 0.11,
                "decision": 0.08,
                "plan_step": 0.07,
                "artifact": 0.05,
                "result": 0.06,
                "summary": 0.03,
            }.get(type_name, 0.0)
            final_score = (
                0.35 * semantic_scores.get(node_id, 0.0)
                + 0.30 * fts_scores.get(node_id, 0.0)
                + 0.15 * graph_scores.get(node_id, 0.0)
                + 0.10 * importance_score
                + 0.08 * confidence_score
                + 0.07 * recency_score
                + 0.03 * graph_degree_score
                + 0.03 * session_match
                + 0.02 * task_match
                + 0.08 * temporal_score
                + type_bias
                + archived_bonus
            ) * status_score
            reasons: List[str] = []
            if node_id in fts_scores:
                reasons.append("fts")
            if node_id in semantic_scores:
                reasons.append("semantic")
            if node_id in graph_scores:
                reasons.append("branch")
            if graph_degree_score > 0:
                reasons.append("graph")
            enriched = dict(row)
            enriched.update(
                {
                    "fts_score": round(fts_scores.get(node_id, 0.0), 6),
                    "semantic_score": round(semantic_scores.get(node_id, 0.0), 6),
                    "branch_score": round(graph_scores.get(node_id, 0.0), 6),
                    "importance_score": round(importance_score, 6),
                    "confidence_score": round(confidence_score, 6),
                    "recency_score": round(recency_score, 6),
                    "temporal_score": round(temporal_score, 6),
                    "graph_score": round(graph_degree_score, 6),
                    "final_score": round(final_score, 6),
                    "retrieval_reasons": reasons,
                }
            )
            ranked.append(enriched)
        ranked.sort(key=lambda row: (row["final_score"], row.get("importance", 0.0), row.get("updated_at") or ""), reverse=True)
        return ranked[:limit]

    def _parse_metadata(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        return {}

    def _find_exact_duplicate(
        self,
        *,
        node_type: str,
        title: str,
        content: str,
        session_id: Optional[str],
        task_id: Optional[str],
        source_kind: Optional[str],
        target: Optional[str] = None,
    ) -> Optional[sqlite3.Row]:
        rows = self._conn.execute(
            """
            SELECT *
            FROM memory_nodes
            WHERE type = ?
              AND status IN ('active', 'resolved')
            ORDER BY updated_at DESC
            """,
            (node_type,),
        ).fetchall()
        want_title = _normalize_text(title)
        want_content = _normalize_text(content)
        for row in rows:
            if session_id and row["session_id"] not in {None, session_id}:
                continue
            if task_id and row["task_id"] not in {None, task_id}:
                continue
            if source_kind and row["source_kind"] not in {None, source_kind}:
                continue
            if _normalize_text(row["title"] or "") != want_title:
                continue
            if _normalize_text(row["content"] or "") != want_content:
                continue
            if target:
                metadata = self._parse_metadata(row["metadata_json"])
                if metadata.get("target") != target:
                    continue
            return row
        return None

    def add_edge(
        self,
        *,
        src_id: str,
        dst_id: str,
        edge_type: str,
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        existing = self._conn.execute(
            """
            SELECT id FROM memory_edges
            WHERE src_id = ? AND dst_id = ? AND edge_type = ?
            LIMIT 1
            """,
            (src_id, dst_id, edge_type),
        ).fetchone()
        now = _utc_now()
        if existing:
            self._conn.execute(
                """
                UPDATE memory_edges
                SET weight = ?, metadata_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    weight,
                    json.dumps(metadata or {}, ensure_ascii=False),
                    now,
                    existing["id"],
                ),
            )
            self._conn.commit()
            return str(existing["id"])

        edge_id = f"edge_{uuid.uuid4().hex}"
        self._conn.execute(
            """
            INSERT INTO memory_edges (
                id, src_id, dst_id, edge_type, weight, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                edge_id,
                src_id,
                dst_id,
                edge_type,
                weight,
                json.dumps(metadata or {}, ensure_ascii=False),
                now,
                now,
            ),
        )
        self._conn.commit()
        return edge_id

    def _archive_node(
        self,
        *,
        node_id: str,
        reason: str,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = _utc_now()
        self._conn.execute(
            """
            UPDATE memory_nodes
            SET status = 'archived', archived_at = ?, updated_at = ?
            WHERE id = ? AND status != 'archived'
            """,
            (now, now, node_id),
        )
        self.record_event(
            event_type="archive",
            node_id=node_id,
            session_id=session_id,
            task_id=task_id,
            reason=reason,
            payload=payload or {},
        )
        self._delete_vec_row(node_id=node_id)

    def _supersede_node(
        self,
        *,
        old_node_id: str,
        new_node_id: str,
        reason: str,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = _utc_now()
        self._conn.execute(
            """
            UPDATE memory_nodes
            SET status = 'superseded', canonical_node_id = ?, updated_at = ?
            WHERE id = ? AND status IN ('active', 'resolved')
            """,
            (new_node_id, now, old_node_id),
        )
        self.add_edge(
            src_id=new_node_id,
            dst_id=old_node_id,
            edge_type="supersedes",
            metadata=payload or {},
        )
        self.record_event(
            event_type="supersede",
            node_id=new_node_id,
            related_node_id=old_node_id,
            session_id=session_id,
            task_id=task_id,
            reason=reason,
            payload=payload or {},
        )
        self._delete_vec_row(node_id=old_node_id)

    def _archive_prior_summaries(
        self,
        *,
        session_id: str,
        keep_node_id: str,
        task_id: Optional[str] = None,
    ) -> None:
        rows = self._conn.execute(
            """
            SELECT id
            FROM memory_nodes
            WHERE session_id = ?
              AND type = 'summary'
              AND id != ?
              AND status IN ('active', 'resolved')
            """,
            (session_id, keep_node_id),
        ).fetchall()
        for row in rows:
            self._archive_node(
                node_id=str(row["id"]),
                reason="replaced_by_newer_session_summary",
                session_id=session_id,
                task_id=task_id,
                payload={"replacement_node_id": keep_node_id},
            )
            self.add_edge(
                src_id=keep_node_id,
                dst_id=str(row["id"]),
                edge_type="summarizes",
                metadata={"reason": "newer_session_summary"},
            )

    def prune_low_value_nodes(
        self,
        *,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        max_nodes: int = 25,
        importance_threshold: float = 0.25,
        confidence_threshold: float = 0.35,
    ) -> List[str]:
        params: List[Any] = [importance_threshold, confidence_threshold]
        sql = """
            SELECT id
            FROM memory_nodes
            WHERE status IN ('archived', 'superseded', 'rejected')
              AND importance <= ?
              AND confidence <= ?
        """
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        if task_id:
            sql += " AND task_id = ?"
            params.append(task_id)
        sql += " ORDER BY updated_at ASC LIMIT ?"
        params.append(max_nodes)
        rows = self._conn.execute(sql, tuple(params)).fetchall()
        pruned_ids: List[str] = []
        now = _utc_now()
        for row in rows:
            node_id = str(row["id"])
            self._conn.execute(
                """
                UPDATE memory_nodes
                SET status = 'pruned', updated_at = ?, archived_at = COALESCE(archived_at, ?)
                WHERE id = ?
                """,
                (now, now, node_id),
            )
            self.record_event(
                event_type="prune",
                node_id=node_id,
                session_id=session_id,
                task_id=task_id,
                reason="low_confidence_low_importance_leaf",
                payload={},
            )
            pruned_ids.append(node_id)
        self._conn.commit()
        return pruned_ids

    def record_event(
        self,
        *,
        event_type: str,
        node_id: Optional[str] = None,
        related_node_id: Optional[str] = None,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        reason: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        event_id = f"evt_{uuid.uuid4().hex}"
        self._conn.execute(
            """
            INSERT INTO memory_events (
                id, event_type, node_id, related_node_id, session_id,
                task_id, reason, payload_json, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                event_type,
                node_id,
                related_node_id,
                session_id,
                task_id,
                reason,
                json.dumps(payload or {}, ensure_ascii=False),
                _utc_now(),
            ),
        )
        self._conn.commit()
        return event_id

    def add_node(
        self,
        *,
        node_type: str,
        content: str,
        title: str = "",
        status: str = "active",
        confidence: float = 0.5,
        importance: float = 0.5,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        source_kind: Optional[str] = None,
        source_ref: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        allow_duplicate: bool = False,
    ) -> str:
        metadata = metadata or {}
        if not allow_duplicate:
            duplicate = self._find_exact_duplicate(
                node_type=node_type,
                title=title,
                content=content,
                session_id=session_id,
                task_id=task_id,
                source_kind=source_kind,
                target=metadata.get("target"),
            )
            if duplicate is not None:
                now = _utc_now()
                merged_confidence = max(float(duplicate["confidence"] or 0.0), confidence)
                merged_importance = max(float(duplicate["importance"] or 0.0), importance)
                self._conn.execute(
                    """
                    UPDATE memory_nodes
                    SET confidence = ?, importance = ?, updated_at = ?, last_accessed_at = ?
                    WHERE id = ?
                    """,
                    (merged_confidence, merged_importance, now, now, duplicate["id"]),
                )
                self.record_event(
                    event_type="merge",
                    node_id=str(duplicate["id"]),
                    session_id=session_id,
                    task_id=task_id,
                    reason="exact_duplicate",
                    payload={
                        "title": title,
                        "source_kind": source_kind,
                        "metadata": metadata,
                    },
                )
                self._conn.commit()
                return str(duplicate["id"])
        node_id = f"mem_{uuid.uuid4().hex}"
        now = _utc_now()
        embed_meta = self._embed_metadata() if self._embedder else {}
        merged_metadata = dict(metadata)
        merged_metadata.update(embed_meta)
        embedding_json = self._maybe_embed(f"{title}\n{content}".strip())
        self._conn.execute(
            """
            INSERT INTO memory_nodes (
                id, type, status, title, content, confidence, importance,
                session_id, task_id, source_kind, source_ref, metadata_json,
                embedding_json, created_at, updated_at, last_accessed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                node_id,
                node_type,
                status,
                title or None,
                content,
                confidence,
                importance,
                session_id,
                task_id,
                source_kind,
                source_ref,
                json.dumps(merged_metadata, ensure_ascii=False),
                embedding_json,
                now,
                now,
                now,
            ),
        )
        self.record_event(
            event_type="create",
            node_id=node_id,
            session_id=session_id,
            task_id=task_id,
            reason=source_kind or "create",
            payload=merged_metadata,
        )
        self._upsert_vec_row(node_id=node_id, vector=self._decode_embedding(embedding_json))
        self._conn.commit()
        return node_id

    def ingest_memory_entry(
        self,
        *,
        target: str,
        content: str,
        action: str = "add",
        old_text: Optional[str] = None,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        source_ref: Optional[str] = None,
    ) -> str:
        node_id = self.add_node(
            node_type="observation" if target == "memory" else "summary",
            title=f"{target.title()} memory",
            content=content,
            confidence=0.8,
            importance=0.7 if target == "user" else 0.55,
            session_id=session_id,
            task_id=task_id,
            source_kind=f"memory_tool:{target}",
            source_ref=source_ref,
            metadata={"target": target, "action": action},
        )
        if action == "replace" and old_text:
            rows = self._conn.execute(
                """
                SELECT id, content
                FROM memory_nodes
                WHERE id != ?
                  AND type = ?
                  AND status IN ('active', 'resolved')
                ORDER BY updated_at DESC
                """,
                (node_id, "observation" if target == "memory" else "summary"),
            ).fetchall()
            needle = _normalize_text(old_text)
            for row in rows:
                metadata = self._parse_metadata(self._conn.execute(
                    "SELECT metadata_json FROM memory_nodes WHERE id = ?",
                    (row["id"],),
                ).fetchone()["metadata_json"])
                if metadata.get("target") != target:
                    continue
                if session_id and self._conn.execute(
                    "SELECT session_id FROM memory_nodes WHERE id = ?",
                    (row["id"],),
                ).fetchone()["session_id"] not in {None, session_id}:
                    continue
                if needle and needle not in _normalize_text(row["content"] or ""):
                    continue
                self._supersede_node(
                    old_node_id=str(row["id"]),
                    new_node_id=node_id,
                    reason="memory_tool_replace",
                    session_id=session_id,
                    task_id=task_id,
                    payload={"target": target, "old_text": old_text},
                )
        self._conn.commit()
        return node_id

    def record_session_goal(self, *, session_id: str, content: str, task_id: Optional[str] = None) -> str:
        existing = self._conn.execute(
            "SELECT id, content FROM memory_nodes WHERE session_id = ? AND type = 'goal' AND status = 'active' ORDER BY created_at LIMIT 1",
            (session_id,),
        ).fetchone()
        if existing:
            if _normalize_text(existing["content"] or "") == _normalize_text(content):
                return str(existing["id"])
        new_id = self.add_node(
            node_type="goal",
            title="Session goal",
            content=content,
            confidence=0.7,
            importance=0.9,
            session_id=session_id,
            task_id=task_id,
            source_kind="session_start",
        )
        if existing:
            self._supersede_node(
                old_node_id=str(existing["id"]),
                new_node_id=new_id,
                reason="updated_session_goal",
                session_id=session_id,
                task_id=task_id,
                payload={"old_content": existing["content"], "new_content": content},
            )
            self._conn.commit()
        return new_id

    def ensure_task_context(self, *, session_id: str, task_id: str, user_message: str) -> Dict[str, str]:
        task_root_id = self.add_node(
            node_type="task",
            title="Active task",
            content=user_message,
            confidence=0.85,
            importance=0.95,
            session_id=session_id,
            task_id=task_id,
            source_kind="task_start",
            source_ref=task_id,
            metadata={"task_id": task_id, "kind": "task_root"},
        )
        goal_id = self.record_session_goal(session_id=session_id, content=user_message, task_id=task_id)
        self.add_edge(
            src_id=task_root_id,
            dst_id=goal_id,
            edge_type="supports",
            metadata={"reason": "task_context"},
        )
        self.record_event(
            event_type="task_context",
            node_id=task_root_id,
            related_node_id=goal_id,
            session_id=session_id,
            task_id=task_id,
            reason="ensure_task_context",
            payload={"task_id": task_id},
        )
        return {"task_root_id": task_root_id, "goal_id": goal_id}

    def _task_context_for(self, *, session_id: str, task_id: str) -> Dict[str, Optional[str]]:
        task_row = self._conn.execute(
            """
            SELECT id
            FROM memory_nodes
            WHERE session_id = ? AND task_id = ? AND type = 'task' AND status IN ('active', 'resolved')
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (session_id, task_id),
        ).fetchone()
        goal_row = self._conn.execute(
            """
            SELECT id
            FROM memory_nodes
            WHERE session_id = ? AND task_id = ? AND type = 'goal' AND status IN ('active', 'resolved')
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (session_id, task_id),
        ).fetchone()
        return {
            "task_root_id": str(task_row["id"]) if task_row else None,
            "goal_id": str(goal_row["id"]) if goal_row else None,
        }

    def begin_tool_step(
        self,
        *,
        session_id: str,
        task_id: str,
        tool_name: str,
        tool_call_id: Optional[str],
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        args = arguments or {}
        preview = _truncate_text(json.dumps(args, ensure_ascii=False, sort_keys=True), 500) or "{}"
        step_id = self.add_node(
            node_type="plan_step",
            title=f"Use tool: {tool_name}",
            content=f"Tool invocation for `{tool_name}` with arguments:\n{preview}",
            confidence=0.72,
            importance=0.62,
            session_id=session_id,
            task_id=task_id,
            source_kind="tool_call",
            source_ref=tool_call_id,
            metadata={
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "tool_arguments": args,
            },
            allow_duplicate=True,
        )
        context = self._task_context_for(session_id=session_id, task_id=task_id)
        if context["task_root_id"]:
            self.add_edge(
                src_id=context["task_root_id"],
                dst_id=step_id,
                edge_type="depends_on",
                metadata={"reason": "tool_step"},
            )
        if context["goal_id"]:
            self.add_edge(
                src_id=step_id,
                dst_id=context["goal_id"],
                edge_type="supports",
                metadata={"reason": "tool_step"},
            )
        self.record_event(
            event_type="tool_step",
            node_id=step_id,
            related_node_id=context["goal_id"],
            session_id=session_id,
            task_id=task_id,
            reason=tool_name,
            payload={"tool_name": tool_name, "tool_call_id": tool_call_id},
        )
        return {"step_id": step_id, "task_root_id": context["task_root_id"], "goal_id": context["goal_id"]}

    def _open_blockers_for_tool(self, *, session_id: str, task_id: str, tool_name: str) -> List[str]:
        rows = self._conn.execute(
            """
            SELECT id
            FROM memory_nodes
            WHERE session_id = ?
              AND task_id = ?
              AND type = 'blocker'
              AND status = 'active'
            ORDER BY updated_at DESC
            """,
            (session_id, task_id),
        ).fetchall()
        blocker_ids: List[str] = []
        for row in rows:
            metadata = self._parse_metadata(
                self._conn.execute(
                    "SELECT metadata_json FROM memory_nodes WHERE id = ?",
                    (row["id"],),
                ).fetchone()["metadata_json"]
            )
            if metadata.get("tool_name") == tool_name:
                blocker_ids.append(str(row["id"]))
        return blocker_ids

    def record_tool_outcome(
        self,
        *,
        session_id: str,
        task_id: str,
        step_id: str,
        goal_id: Optional[str],
        task_root_id: Optional[str],
        tool_name: str,
        tool_call_id: Optional[str],
        arguments: Optional[Dict[str, Any]],
        result_text: str,
        success: bool,
        duration_seconds: Optional[float] = None,
    ) -> Dict[str, Optional[str]]:
        args = arguments or {}
        excerpt = _truncate_text(result_text, 2000)
        payload = {
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "duration_seconds": round(float(duration_seconds or 0.0), 6),
            "tool_arguments": args,
        }
        if success:
            artifact_id = self.add_node(
                node_type="artifact",
                title=f"{tool_name} output",
                content=excerpt or f"{tool_name} completed successfully.",
                confidence=0.72,
                importance=0.6,
                session_id=session_id,
                task_id=task_id,
                source_kind=f"tool_result:{tool_name}",
                source_ref=tool_call_id,
                metadata=payload,
                allow_duplicate=True,
            )
            self.add_edge(src_id=step_id, dst_id=artifact_id, edge_type="produced", metadata={"tool_name": tool_name})
            if goal_id:
                self.add_edge(src_id=artifact_id, dst_id=goal_id, edge_type="supports", metadata={"tool_name": tool_name})
            if task_root_id:
                self.add_edge(src_id=task_root_id, dst_id=artifact_id, edge_type="derived_from", metadata={"tool_name": tool_name})
            for blocker_id in self._open_blockers_for_tool(session_id=session_id, task_id=task_id, tool_name=tool_name):
                now = _utc_now()
                self._conn.execute(
                    """
                    UPDATE memory_nodes
                    SET status = 'resolved', resolved_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (now, now, blocker_id),
                )
                self.add_edge(src_id=artifact_id, dst_id=blocker_id, edge_type="resolves", metadata={"tool_name": tool_name})
                self.record_event(
                    event_type="resolve",
                    node_id=artifact_id,
                    related_node_id=blocker_id,
                    session_id=session_id,
                    task_id=task_id,
                    reason=f"tool_success:{tool_name}",
                    payload=payload,
                )
            self.record_event(
                event_type="tool_success",
                node_id=artifact_id,
                related_node_id=step_id,
                session_id=session_id,
                task_id=task_id,
                reason=tool_name,
                payload=payload,
            )
            self._conn.commit()
            return {"node_id": artifact_id}

        blocker_id = self.add_node(
            node_type="blocker",
            title=f"{tool_name} failure",
            content=excerpt or f"{tool_name} failed.",
            confidence=0.88,
            importance=0.82,
            session_id=session_id,
            task_id=task_id,
            source_kind=f"tool_error:{tool_name}",
            source_ref=tool_call_id,
            metadata=payload,
            allow_duplicate=True,
        )
        self.add_edge(src_id=blocker_id, dst_id=step_id, edge_type="blocks", metadata={"tool_name": tool_name})
        if goal_id:
            self.add_edge(src_id=blocker_id, dst_id=goal_id, edge_type="blocks", metadata={"tool_name": tool_name})
        if task_root_id:
            self.add_edge(src_id=task_root_id, dst_id=blocker_id, edge_type="depends_on", metadata={"tool_name": tool_name})
        self.record_event(
            event_type="tool_error",
            node_id=blocker_id,
            related_node_id=step_id,
            session_id=session_id,
            task_id=task_id,
            reason=tool_name,
            payload=payload,
        )
        self._conn.commit()
        return {"node_id": blocker_id}

    def archive_session_summary(self, *, session_id: str, summary_text: str, task_id: Optional[str] = None) -> Optional[str]:
        summary = summary_text.strip()
        if not summary:
            return None
        node_id = self.add_node(
            node_type="summary",
            title="Compressed context summary",
            content=summary,
            confidence=0.75,
            importance=0.6,
            session_id=session_id,
            task_id=task_id,
            source_kind="context_compression",
        )
        self._archive_prior_summaries(session_id=session_id, keep_node_id=node_id, task_id=task_id)
        self.record_event(
            event_type="summarize",
            node_id=node_id,
            session_id=session_id,
            task_id=task_id,
            reason="context_compression",
        )
        return node_id

    def _summarize_resolved_branch(
        self,
        *,
        session_id: Optional[str],
        task_id: Optional[str],
        stale_before: str,
        max_members: int = 6,
    ) -> Optional[str]:
        params: List[Any] = [stale_before]
        sql = """
            SELECT *
            FROM memory_nodes
            WHERE (
                    (status IN ('resolved', 'archived') AND type IN ('artifact', 'result', 'blocker', 'plan_step', 'decision'))
                 OR (status = 'active' AND type IN ('artifact', 'result', 'plan_step', 'decision'))
                  )
              AND COALESCE(resolved_at, archived_at, updated_at, created_at) <= ?
        """
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        if task_id:
            sql += " AND task_id = ?"
            params.append(task_id)
        sql += """
            ORDER BY importance DESC, updated_at DESC
            LIMIT ?
        """
        params.append(max_members)
        rows = [dict(row) for row in self._conn.execute(sql, tuple(params)).fetchall()]
        if len(rows) < 2:
            return None

        active_blocker = self._conn.execute(
            """
            SELECT 1
            FROM memory_nodes
            WHERE type = 'blocker'
              AND status = 'active'
              AND (? IS NULL OR session_id = ?)
              AND (? IS NULL OR task_id = ?)
            LIMIT 1
            """,
            (session_id, session_id, task_id, task_id),
        ).fetchone()
        if active_blocker:
            return None

        lines = []
        for row in rows:
            label = row.get("title") or row.get("type") or "memory"
            body = _truncate_text(str(row.get("content") or ""), 180).replace("\n", " ")
            lines.append(f"- [{row.get('type')}|{row.get('status')}] {label}: {body}")
        summary_text = "Resolved branch summary:\n" + "\n".join(lines)
        summary_id = self.add_node(
            node_type="summary",
            title="Resolved branch summary",
            content=summary_text,
            confidence=0.78,
            importance=0.58,
            session_id=session_id,
            task_id=task_id,
            source_kind="maintenance_summary",
            metadata={"member_count": len(rows)},
            allow_duplicate=True,
        )
        for row in rows:
            self.add_edge(
                src_id=summary_id,
                dst_id=str(row["id"]),
                edge_type="summarizes",
                metadata={"reason": "maintenance"},
            )
            if str(row.get("status") or "") != "archived":
                self._archive_node(
                    node_id=str(row["id"]),
                    reason="summarized_resolved_branch",
                    session_id=session_id,
                    task_id=task_id,
                    payload={"summary_node_id": summary_id},
                )
        self.record_event(
            event_type="maintenance_summary",
            node_id=summary_id,
            session_id=session_id,
            task_id=task_id,
            reason="resolved_branch",
            payload={"member_ids": [str(row["id"]) for row in rows]},
        )
        self._conn.commit()
        return summary_id

    def _archive_stale_low_value_nodes(
        self,
        *,
        session_id: Optional[str],
        task_id: Optional[str],
        stale_before: str,
        max_nodes: int = 25,
        importance_threshold: float = 0.45,
        confidence_threshold: float = 0.55,
    ) -> List[str]:
        params: List[Any] = [stale_before, importance_threshold, confidence_threshold]
        sql = """
            SELECT n.id
            FROM memory_nodes n
            WHERE n.status IN ('active', 'resolved')
              AND n.type IN ('artifact', 'plan_step', 'observation', 'result')
              AND COALESCE(n.updated_at, n.created_at) <= ?
              AND n.importance <= ?
              AND n.confidence <= ?
        """
        if session_id:
            sql += " AND n.session_id = ?"
            params.append(session_id)
        if task_id:
            sql += " AND n.task_id = ?"
            params.append(task_id)
        sql += """
            ORDER BY n.updated_at ASC
            LIMIT ?
        """
        params.append(max_nodes)
        rows = self._conn.execute(sql, tuple(params)).fetchall()
        archived_ids: List[str] = []
        for row in rows:
            node_id = str(row["id"])
            edge_row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM memory_edges WHERE src_id = ? OR dst_id = ?",
                (node_id, node_id),
            ).fetchone()
            if edge_row and int(edge_row["c"] or 0) > 2:
                continue
            self._archive_node(
                node_id=node_id,
                reason="stale_low_value_leaf",
                session_id=session_id,
                task_id=task_id,
                payload={},
            )
            archived_ids.append(node_id)
        self._conn.commit()
        return archived_ids

    def _merge_semantic_pair(
        self,
        *,
        canonical_id: str,
        duplicate_id: str,
        similarity: float,
        session_id: Optional[str],
        task_id: Optional[str],
    ) -> None:
        canonical = self._conn.execute(
            "SELECT * FROM memory_nodes WHERE id = ?",
            (canonical_id,),
        ).fetchone()
        duplicate = self._conn.execute(
            "SELECT * FROM memory_nodes WHERE id = ?",
            (duplicate_id,),
        ).fetchone()
        if not canonical or not duplicate:
            return
        merged_confidence = max(float(canonical["confidence"] or 0.0), float(duplicate["confidence"] or 0.0), min(0.99, similarity))
        merged_importance = max(float(canonical["importance"] or 0.0), float(duplicate["importance"] or 0.0))
        canonical_meta = self._parse_metadata(canonical["metadata_json"])
        duplicate_meta = self._parse_metadata(duplicate["metadata_json"])
        merged_meta = dict(canonical_meta)
        merged_meta.setdefault("semantic_merge_sources", [])
        merged_meta["semantic_merge_sources"] = list(dict.fromkeys(
            list(merged_meta["semantic_merge_sources"]) + [duplicate_id]
        ))
        for key, value in duplicate_meta.items():
            merged_meta.setdefault(key, value)
        now = _utc_now()
        self._conn.execute(
            """
            UPDATE memory_nodes
            SET confidence = ?, importance = ?, metadata_json = ?, updated_at = ?, last_accessed_at = ?
            WHERE id = ?
            """,
            (
                merged_confidence,
                merged_importance,
                json.dumps(merged_meta, ensure_ascii=False),
                now,
                now,
                canonical_id,
            ),
        )
        self._supersede_node(
            old_node_id=duplicate_id,
            new_node_id=canonical_id,
            reason="semantic_high_confidence",
            session_id=session_id,
            task_id=task_id,
            payload={"similarity": similarity},
        )
        self.record_event(
            event_type="merge",
            node_id=canonical_id,
            related_node_id=duplicate_id,
            session_id=session_id,
            task_id=task_id,
            reason="semantic_high_confidence",
            payload={"similarity": similarity},
        )

    def _existing_hypothesis_for_sources(self, *, source_ids: List[str]) -> Optional[str]:
        wanted = set(source_ids)
        rows = self._conn.execute(
            """
            SELECT id, metadata_json
            FROM memory_nodes
            WHERE type = 'hypothesis'
              AND status IN ('active', 'resolved')
            ORDER BY updated_at DESC
            """
        ).fetchall()
        for row in rows:
            metadata = self._parse_metadata(row["metadata_json"])
            existing = set(str(x) for x in metadata.get("source_ids", []) if x)
            if existing == wanted:
                return str(row["id"])
        return None

    def synthesize_hypotheses(
        self,
        *,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        candidate_min_similarity: float = 0.58,
        candidate_max_similarity: float = 0.94,
        max_hypotheses: int = 5,
    ) -> List[str]:
        params: List[Any] = []
        sql = """
            SELECT *
            FROM memory_nodes
            WHERE status IN ('active', 'resolved', 'archived')
              AND type IN ('observation', 'artifact', 'summary', 'result')
              AND embedding_json IS NOT NULL
        """
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        if task_id:
            sql += " AND (task_id = ? OR task_id IS NULL)"
            params.append(task_id)
        sql += " ORDER BY updated_at DESC LIMIT 40"
        rows = [dict(row) for row in self._conn.execute(sql, tuple(params)).fetchall()]
        prepared: List[Dict[str, Any]] = []
        for row in rows:
            metadata = self._parse_metadata(row.get("metadata_json"))
            vector = self._decode_embedding(row.get("embedding_json"))
            if not vector or not self._embedding_compatible(metadata, vector, len(vector)):
                continue
            row["_vector"] = vector
            prepared.append(row)

        created: List[str] = []
        for idx, left in enumerate(prepared):
            if len(created) >= max_hypotheses:
                break
            for right in prepared[idx + 1:]:
                if len(created) >= max_hypotheses:
                    break
                if left["type"] == right["type"] and left.get("task_id") == right.get("task_id"):
                    continue
                similarity = _cosine_similarity(left["_vector"], right["_vector"])
                if similarity < candidate_min_similarity or similarity > candidate_max_similarity:
                    continue
                left_tokens = set(_significant_tokens(f"{left.get('title') or ''} {left.get('content') or ''}"))
                right_tokens = set(_significant_tokens(f"{right.get('title') or ''} {right.get('content') or ''}"))
                overlap = [tok for tok in left_tokens & right_tokens if len(tok) >= 4][:4]
                if not overlap:
                    continue
                source_ids = [str(left["id"]), str(right["id"])]
                if self._existing_hypothesis_for_sources(source_ids=source_ids):
                    continue
                left_label = left.get("title") or left.get("type") or "memory"
                right_label = right.get("title") or right.get("type") or "memory"
                hypothesis_text = (
                    f"Possible synthesis: `{left_label}` and `{right_label}` may reflect the same underlying pattern "
                    f"around {', '.join(overlap[:3])}. A safe test could inspect related files, search prior sessions, "
                    "or run read-only terminal checks to confirm or refute the connection."
                )
                hypothesis_id = self.add_node(
                    node_type="hypothesis",
                    title="Synthesized hypothesis",
                    content=hypothesis_text,
                    confidence=min(0.82, max(0.52, similarity)),
                    importance=0.67,
                    session_id=session_id,
                    task_id=task_id,
                    source_kind="memory_synthesis",
                    metadata={
                        "source_ids": source_ids,
                        "predicted_terms": overlap,
                        "similarity": similarity,
                        "test_strategy": "safe_read_only_validation",
                    },
                    allow_duplicate=True,
                )
                self.add_edge(src_id=source_ids[0], dst_id=hypothesis_id, edge_type="derived_from", weight=similarity, metadata={"similarity": similarity})
                self.add_edge(src_id=source_ids[1], dst_id=hypothesis_id, edge_type="derived_from", weight=similarity, metadata={"similarity": similarity})
                self.record_event(
                    event_type="hypothesis_create",
                    node_id=hypothesis_id,
                    session_id=session_id,
                    task_id=task_id,
                    reason="memory_synthesis",
                    payload={"source_ids": source_ids, "predicted_terms": overlap, "similarity": similarity},
                )
                created.append(hypothesis_id)
        self._conn.commit()
        return created

    def evaluate_hypotheses_from_tool(
        self,
        *,
        session_id: str,
        task_id: str,
        tool_name: str,
        result_text: str,
        artifact_id: Optional[str] = None,
        safe_test: bool = False,
    ) -> List[str]:
        if not safe_test or not result_text.strip():
            return []
        params: List[Any] = [session_id, task_id]
        rows = self._conn.execute(
            """
            SELECT *
            FROM memory_nodes
            WHERE type = 'hypothesis'
              AND status = 'active'
              AND session_id = ?
              AND (task_id = ? OR task_id IS NULL)
            ORDER BY updated_at DESC
            """,
            tuple(params),
        ).fetchall()
        normalized_result = _normalize_text(result_text)
        negative = bool(re.search(r"\b(not found|no such|missing|failed|does not exist|cannot|error)\b", normalized_result))
        created_results: List[str] = []
        for row in rows:
            metadata = self._parse_metadata(row["metadata_json"])
            predicted_terms = [str(term).lower() for term in metadata.get("predicted_terms", []) if str(term).strip()]
            if not predicted_terms:
                continue
            hits = [term for term in predicted_terms if term in normalized_result]
            if hits:
                verdict = "supported"
                edge_type = "supports"
            elif negative:
                verdict = "contradicted"
                edge_type = "contradicts"
            else:
                continue
            result_id = self.add_node(
                node_type="result",
                title=f"Hypothesis {verdict}",
                content=_truncate_text(
                    f"Hypothesis evaluation via {tool_name}: {verdict}. Evidence terms: {', '.join(hits) or 'negative result markers'}.\n\n"
                    f"Tool output:\n{result_text}",
                    2400,
                ),
                confidence=0.74 if verdict == "supported" else 0.68,
                importance=0.7,
                session_id=session_id,
                task_id=task_id,
                source_kind=f"hypothesis_test:{tool_name}",
                metadata={
                    "hypothesis_id": str(row["id"]),
                    "verdict": verdict,
                    "evidence_terms": hits,
                    "safe_test": safe_test,
                },
                allow_duplicate=True,
            )
            self.add_edge(src_id=result_id, dst_id=str(row["id"]), edge_type=edge_type, metadata={"tool_name": tool_name, "evidence_terms": hits})
            if artifact_id:
                self.add_edge(src_id=artifact_id, dst_id=result_id, edge_type="derived_from", metadata={"tool_name": tool_name})
            now = _utc_now()
            self._conn.execute(
                """
                UPDATE memory_nodes
                SET status = 'resolved', resolved_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (now, now, row["id"]),
            )
            self.record_event(
                event_type="hypothesis_result",
                node_id=result_id,
                related_node_id=str(row["id"]),
                session_id=session_id,
                task_id=task_id,
                reason=verdict,
                payload={"tool_name": tool_name, "evidence_terms": hits, "safe_test": safe_test},
            )
            created_results.append(result_id)
        self._conn.commit()
        return created_results

    def _semantic_merge_candidates(
        self,
        *,
        session_id: Optional[str],
        task_id: Optional[str],
        candidate_threshold: float = 0.92,
        auto_merge_threshold: float = 0.975,
        max_pairs: int = 40,
    ) -> Dict[str, Any]:
        low_risk_auto_types = {"observation", "summary", "artifact"}
        protected_types = {"goal", "task", "blocker", "decision", "plan_step"}
        params: List[Any] = []
        sql = """
            SELECT *
            FROM memory_nodes
            WHERE status IN ('active', 'resolved')
              AND embedding_json IS NOT NULL
        """
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        if task_id:
            sql += " AND task_id = ?"
            params.append(task_id)
        sql += " ORDER BY updated_at DESC"
        rows = [dict(row) for row in self._conn.execute(sql, tuple(params)).fetchall()]
        prepared: List[Dict[str, Any]] = []
        for row in rows:
            metadata = self._parse_metadata(row.get("metadata_json"))
            vector = self._decode_embedding(row.get("embedding_json"))
            if not self._embedding_compatible(metadata, vector, len(vector or [])):
                continue
            if not vector:
                continue
            row["_vector"] = vector
            prepared.append(row)

        candidate_links: List[Dict[str, Any]] = []
        auto_merged: List[Dict[str, Any]] = []
        seen_pairs = set()
        for idx, left in enumerate(prepared):
            for right in prepared[idx + 1:]:
                if len(candidate_links) + len(auto_merged) >= max_pairs:
                    break
                if left["type"] != right["type"]:
                    continue
                pair = tuple(sorted((str(left["id"]), str(right["id"]))))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                similarity = _cosine_similarity(left["_vector"], right["_vector"])
                if similarity < candidate_threshold:
                    continue
                left_type = str(left["type"])
                left_time = _node_time(left) or 0.0
                right_time = _node_time(right) or 0.0
                newer = left if left_time >= right_time else right
                older = right if newer is left else left
                if left_type in low_risk_auto_types and similarity >= auto_merge_threshold:
                    self._merge_semantic_pair(
                        canonical_id=str(newer["id"]),
                        duplicate_id=str(older["id"]),
                        similarity=similarity,
                        session_id=session_id,
                        task_id=task_id,
                    )
                    auto_merged.append({"canonical_id": str(newer["id"]), "merged_id": str(older["id"]), "similarity": round(similarity, 6)})
                    continue
                edge_type = "merge_candidate" if left_type in protected_types else "similar_to"
                self.add_edge(
                    src_id=str(newer["id"]),
                    dst_id=str(older["id"]),
                    edge_type=edge_type,
                    weight=similarity,
                    metadata={"similarity": similarity, "semantic_candidate": True},
                )
                self.record_event(
                    event_type="merge_candidate",
                    node_id=str(newer["id"]),
                    related_node_id=str(older["id"]),
                    session_id=session_id,
                    task_id=task_id,
                    reason="semantic_similarity",
                    payload={"similarity": similarity, "edge_type": edge_type},
                )
                candidate_links.append({"src_id": str(newer["id"]), "dst_id": str(older["id"]), "edge_type": edge_type, "similarity": round(similarity, 6)})
            if len(candidate_links) + len(auto_merged) >= max_pairs:
                break

        self._conn.commit()
        return {"auto_merged": auto_merged, "candidate_links": candidate_links}

    def run_maintenance(
        self,
        *,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        stale_days: int = 30,
        prune_max: int = 25,
    ) -> Dict[str, Any]:
        stale_before = _dt_to_utc_text(_utc_now_dt() - timedelta(days=max(1, stale_days)))
        summary_id = self._summarize_resolved_branch(
            session_id=session_id,
            task_id=task_id,
            stale_before=stale_before,
        )
        semantic_merge = self._semantic_merge_candidates(
            session_id=session_id,
            task_id=task_id,
        )
        synthesized_hypotheses = self.synthesize_hypotheses(
            session_id=session_id,
            task_id=task_id,
        )
        archived_ids = self._archive_stale_low_value_nodes(
            session_id=session_id,
            task_id=task_id,
            stale_before=stale_before,
        )
        pruned_ids = self.prune_low_value_nodes(
            session_id=session_id,
            task_id=task_id,
            max_nodes=prune_max,
        )
        result = {
            "summary_id": summary_id,
            "semantic_merge": semantic_merge,
            "synthesized_hypotheses": synthesized_hypotheses,
            "archived_ids": archived_ids,
            "pruned_ids": pruned_ids,
            "stale_before": stale_before,
        }
        self.record_event(
            event_type="maintenance",
            session_id=session_id,
            task_id=task_id,
            reason="manual_or_exit_sweep",
            payload={
                "summary_id": summary_id,
                "auto_merged_count": len(semantic_merge["auto_merged"]),
                "merge_candidate_count": len(semantic_merge["candidate_links"]),
                "hypothesis_count": len(synthesized_hypotheses),
                "archived_count": len(archived_ids),
                "pruned_count": len(pruned_ids),
                "stale_before": stale_before,
            },
        )
        self._conn.commit()
        return result

    def retrieve_frontier(
        self,
        *,
        query: str = "",
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        time_from: Optional[str] = None,
        time_to: Optional[str] = None,
        prefer_recent: bool = False,
        prefer_archived: bool = False,
        prefer_older: bool = False,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        temporal = _extract_temporal_preferences(query)
        clean_query = temporal["clean_query"] or query
        effective_time_from = time_from or temporal.get("time_from")
        effective_time_to = time_to or temporal.get("time_to")
        effective_prefer_recent = prefer_recent or bool(temporal.get("prefer_recent"))
        effective_prefer_archived = prefer_archived or bool(temporal.get("prefer_archived"))
        effective_prefer_older = prefer_older or bool(temporal.get("prefer_older"))

        branch_seeds = self._active_branch_seed_ids(
            session_id=session_id,
            task_id=task_id,
            time_from=effective_time_from,
            time_to=effective_time_to,
            include_archived=effective_prefer_archived,
            limit=max(limit, 6),
        )
        if clean_query.strip():
            lexical = self._fts_candidates(
                query=clean_query,
                session_id=session_id,
                task_id=task_id,
                time_from=effective_time_from,
                time_to=effective_time_to,
                include_archived=effective_prefer_archived,
                limit=max(limit * 4, 12),
            )
            if not lexical:
                lexical = self._fallback_text_candidates(
                    query=clean_query,
                    session_id=session_id,
                    task_id=task_id,
                    time_from=effective_time_from,
                    time_to=effective_time_to,
                    include_archived=effective_prefer_archived,
                    limit=max(limit * 4, 12),
                )
            semantic = self._semantic_candidates(
                query=clean_query,
                session_id=session_id,
                task_id=task_id,
                time_from=effective_time_from,
                time_to=effective_time_to,
                include_archived=effective_prefer_archived,
                limit=max(limit * 4, 12),
            )
            dynamic_seed_ids = branch_seeds + [str(row["id"]) for row in lexical[:3]] + [str(row["id"]) for row in semantic[:3]]
            graph = self._graph_neighbor_candidates(
                seed_ids=list(dict.fromkeys(dynamic_seed_ids)),
                session_id=session_id,
                task_id=task_id,
                time_from=effective_time_from,
                time_to=effective_time_to,
                include_archived=effective_prefer_archived,
                limit=max(limit * 4, 12),
            )
            out = self._fuse_candidates(
                lexical=lexical,
                semantic=semantic,
                graph=graph,
                session_id=session_id,
                task_id=task_id,
                prefer_recent=effective_prefer_recent,
                prefer_older=effective_prefer_older,
                include_archived=effective_prefer_archived,
                limit=limit,
            )
        else:
            graph = self._graph_neighbor_candidates(
                seed_ids=branch_seeds,
                session_id=session_id,
                task_id=task_id,
                time_from=effective_time_from,
                time_to=effective_time_to,
                include_archived=effective_prefer_archived,
                limit=max(limit * 3, 12),
            )
            out = self._fuse_candidates(
                lexical=[],
                semantic=[],
                graph=graph,
                session_id=session_id,
                task_id=task_id,
                prefer_recent=effective_prefer_recent,
                prefer_older=effective_prefer_older,
                include_archived=effective_prefer_archived,
                limit=limit,
            )
        now = _utc_now()
        for row in out:
            self._conn.execute(
                "UPDATE memory_nodes SET last_accessed_at = ? WHERE id = ?",
                (now, row["id"]),
            )
            self.record_event(
                event_type="retrieve",
                node_id=row["id"],
                session_id=session_id,
                task_id=task_id,
                reason="frontier_retrieval",
                payload={
                    "query": query,
                    "clean_query": clean_query,
                    "time_from": effective_time_from,
                    "time_to": effective_time_to,
                    "prefer_recent": effective_prefer_recent,
                    "prefer_archived": effective_prefer_archived,
                    "prefer_older": effective_prefer_older,
                },
            )
        self._conn.commit()
        return out

    def retrieve_working_memory(
        self,
        *,
        query: str = "",
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 6,
    ) -> List[Dict[str, Any]]:
        rows = self.retrieve_frontier(
            query=query,
            session_id=session_id,
            task_id=task_id,
            limit=limit,
        )
        packet: List[Dict[str, Any]] = []
        for row in rows:
            brief = _truncate_text(str(row.get("content") or ""), 220).replace("\n", " ")
            reasons = row.get("retrieval_reasons") or []
            packet.append(
                {
                    "node_id": str(row["id"]),
                    "type": str(row.get("type") or ""),
                    "status": str(row.get("status") or ""),
                    "title": str(row.get("title") or row.get("type") or "memory"),
                    "brief": brief,
                    "why_relevant": ", ".join(reasons) if reasons else "relevance",
                    "score": float(row.get("final_score", 0.0) or 0.0),
                    "session_id": row.get("session_id"),
                    "task_id": row.get("task_id"),
                    "source_ids": self._parse_metadata(row.get("metadata_json")).get("source_ids", []),
                }
            )
        return packet

    def expand_memory_node(
        self,
        node_id: str,
        *,
        include_neighbors: bool = True,
        include_events: bool = True,
        neighbor_limit: int = 8,
        event_limit: int = 8,
    ) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM memory_nodes WHERE id = ?",
            (node_id,),
        ).fetchone()
        if not row:
            return None
        node = dict(row)
        node["metadata"] = self._parse_metadata(node.get("metadata_json"))
        node.pop("metadata_json", None)

        out: Dict[str, Any] = {"node": node}

        if include_neighbors:
            neighbors = self._conn.execute(
                """
                SELECT
                    e.id AS edge_id,
                    e.edge_type,
                    e.weight,
                    e.metadata_json,
                    e.created_at AS edge_created_at,
                    CASE WHEN e.src_id = ? THEN e.dst_id ELSE e.src_id END AS related_id,
                    n.type,
                    n.status,
                    n.title,
                    n.content,
                    n.updated_at
                FROM memory_edges e
                JOIN memory_nodes n
                  ON n.id = CASE WHEN e.src_id = ? THEN e.dst_id ELSE e.src_id END
                WHERE e.src_id = ? OR e.dst_id = ?
                ORDER BY e.updated_at DESC, n.updated_at DESC
                LIMIT ?
                """,
                (node_id, node_id, node_id, node_id, neighbor_limit),
            ).fetchall()
            out["neighbors"] = [
                {
                    "edge_id": str(nei["edge_id"]),
                    "edge_type": str(nei["edge_type"]),
                    "weight": float(nei["weight"] or 0.0),
                    "edge_metadata": self._parse_metadata(nei["metadata_json"]),
                    "related_id": str(nei["related_id"]),
                    "type": str(nei["type"] or ""),
                    "status": str(nei["status"] or ""),
                    "title": str(nei["title"] or nei["type"] or "memory"),
                    "brief": _truncate_text(str(nei["content"] or ""), 220).replace("\n", " "),
                    "updated_at": nei["updated_at"],
                }
                for nei in neighbors
            ]

        if include_events:
            events = self._conn.execute(
                """
                SELECT *
                FROM memory_events
                WHERE node_id = ? OR related_node_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (node_id, node_id, event_limit),
            ).fetchall()
            out["events"] = [
                {
                    "id": str(evt["id"]),
                    "event_type": str(evt["event_type"]),
                    "node_id": evt["node_id"],
                    "related_node_id": evt["related_node_id"],
                    "reason": str(evt["reason"] or ""),
                    "payload": self._parse_metadata(evt["payload_json"]),
                    "timestamp": evt["timestamp"],
                }
                for evt in events
            ]

        return out

    def list_nodes(
        self,
        *,
        node_type: Optional[str] = None,
        status: Optional[str] = None,
        source_kind: Optional[str] = None,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        params: List[Any] = []
        sql = "SELECT * FROM memory_nodes WHERE 1=1"
        if node_type:
            sql += " AND type = ?"
            params.append(node_type)
        if status:
            sql += " AND status = ?"
            params.append(status)
        if source_kind:
            sql += " AND source_kind = ?"
            params.append(source_kind)
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        if task_id:
            sql += " AND task_id = ?"
            params.append(task_id)
        sql += " ORDER BY updated_at DESC, created_at DESC LIMIT ?"
        params.append(limit)
        return [dict(row) for row in self._conn.execute(sql, tuple(params)).fetchall()]

    def list_events(
        self,
        *,
        event_type: Optional[str] = None,
        reason_prefix: Optional[str] = None,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        params: List[Any] = []
        sql = "SELECT * FROM memory_events WHERE 1=1"
        if event_type:
            sql += " AND event_type = ?"
            params.append(event_type)
        if reason_prefix:
            sql += " AND reason LIKE ?"
            params.append(f"{reason_prefix}%")
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        if task_id:
            sql += " AND task_id = ?"
            params.append(task_id)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        return [dict(row) for row in self._conn.execute(sql, tuple(params)).fetchall()]

    def memory_dashboard(
        self,
        *,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, int]:
        def _count(where: str, params: List[Any]) -> int:
            row = self._conn.execute(
                f"SELECT COUNT(*) AS count FROM memory_nodes WHERE {where}",
                tuple(params),
            ).fetchone()
            return int(row["count"] if row else 0)

        scope_params: List[Any] = []
        scope_clauses: List[str] = ["1=1"]
        if session_id:
            scope_clauses.append("session_id = ?")
            scope_params.append(session_id)
        if task_id:
            scope_clauses.append("task_id = ?")
            scope_params.append(task_id)
        scope = " AND ".join(scope_clauses)

        event_params: List[Any] = []
        event_scope_clauses: List[str] = ["1=1"]
        if session_id:
            event_scope_clauses.append("session_id = ?")
            event_params.append(session_id)
        if task_id:
            event_scope_clauses.append("task_id = ?")
            event_params.append(task_id)
        event_scope = " AND ".join(event_scope_clauses)

        merge_row = self._conn.execute(
            f"SELECT COUNT(*) AS count FROM memory_events WHERE event_type = 'merge_candidate' AND {event_scope}",
            tuple(event_params),
        ).fetchone()

        return {
            "total_nodes": _count(scope, list(scope_params)),
            "active_goals": _count(f"type = 'goal' AND status = 'active' AND {scope}", list(scope_params)),
            "active_blockers": _count(f"type = 'blocker' AND status = 'active' AND {scope}", list(scope_params)),
            "active_hypotheses": _count(f"type = 'hypothesis' AND status = 'active' AND {scope}", list(scope_params)),
            "dream_artifacts": _count(
                f"source_kind IN ('dreamcycle_hypothesis', 'dreamcycle_experiment', 'dreamcycle_prune_candidate') AND {scope}",
                list(scope_params),
            ),
            "prune_candidates": _count(
                f"source_kind = 'dreamcycle_prune_candidate' AND status IN ('active', 'resolved') AND {scope}",
                list(scope_params),
            ),
            "merge_candidates": int(merge_row["count"] if merge_row else 0),
        }

    def review_prune_candidate(
        self,
        *,
        candidate_id: str,
        action: str,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> bool:
        row = self._conn.execute(
            "SELECT * FROM memory_nodes WHERE id = ? AND source_kind = 'dreamcycle_prune_candidate' LIMIT 1",
            (candidate_id,),
        ).fetchone()
        if not row:
            return False
        metadata = self._parse_metadata(row["metadata_json"])
        target_node_id = str(metadata.get("target_node_id") or "")
        if not target_node_id:
            return False
        normalized_action = action.strip().lower()
        now = _utc_now()

        if normalized_action in {"approve", "archive"}:
            self._archive_node(
                node_id=target_node_id,
                reason="user_approved_prune_candidate_archive",
                session_id=session_id or row["session_id"],
                task_id=task_id or row["task_id"],
                payload={"candidate_id": candidate_id},
            )
            self._conn.execute(
                """
                UPDATE memory_nodes
                SET status = 'resolved', resolved_at = COALESCE(resolved_at, ?), updated_at = ?
                WHERE id = ?
                """,
                (now, now, candidate_id),
            )
            self.record_event(
                event_type="review",
                node_id=candidate_id,
                related_node_id=target_node_id,
                session_id=session_id or row["session_id"],
                task_id=task_id or row["task_id"],
                reason="prune_candidate_approved",
                payload={"action": "archive"},
            )
        elif normalized_action in {"reject", "keep"}:
            self._conn.execute(
                """
                UPDATE memory_nodes
                SET status = 'rejected', updated_at = ?
                WHERE id = ?
                """,
                (now, candidate_id),
            )
            self.record_event(
                event_type="review",
                node_id=candidate_id,
                related_node_id=target_node_id,
                session_id=session_id or row["session_id"],
                task_id=task_id or row["task_id"],
                reason="prune_candidate_rejected",
                payload={"action": "keep"},
            )
        else:
            return False
        self._conn.commit()
        return True

    def review_merge_candidate(
        self,
        *,
        canonical_node_id: str,
        duplicate_node_id: str,
        action: str,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> bool:
        left = self._conn.execute("SELECT * FROM memory_nodes WHERE id = ? LIMIT 1", (canonical_node_id,)).fetchone()
        right = self._conn.execute("SELECT * FROM memory_nodes WHERE id = ? LIMIT 1", (duplicate_node_id,)).fetchone()
        if not left or not right:
            return False
        normalized_action = action.strip().lower()
        payload = {"user_review": True}

        if normalized_action in {"approve", "merge"}:
            self._supersede_node(
                old_node_id=duplicate_node_id,
                new_node_id=canonical_node_id,
                reason="user_approved_merge_candidate",
                session_id=session_id or left["session_id"] or right["session_id"],
                task_id=task_id or left["task_id"] or right["task_id"],
                payload=payload,
            )
            self.record_event(
                event_type="review",
                node_id=canonical_node_id,
                related_node_id=duplicate_node_id,
                session_id=session_id or left["session_id"] or right["session_id"],
                task_id=task_id or left["task_id"] or right["task_id"],
                reason="merge_candidate_approved",
                payload=payload,
            )
        elif normalized_action in {"reject", "keep-separate"}:
            self.add_edge(
                src_id=canonical_node_id,
                dst_id=duplicate_node_id,
                edge_type="merge_rejected",
                metadata=payload,
            )
            self.record_event(
                event_type="review",
                node_id=canonical_node_id,
                related_node_id=duplicate_node_id,
                session_id=session_id or left["session_id"] or right["session_id"],
                task_id=task_id or left["task_id"] or right["task_id"],
                reason="merge_candidate_rejected",
                payload=payload,
            )
        else:
            return False
        self._conn.commit()
        return True
