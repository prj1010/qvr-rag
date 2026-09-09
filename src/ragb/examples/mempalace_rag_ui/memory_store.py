"""DuckDB-backed workspace memory for a single local instance.

Stores conversation turns as separate question/answer rows, a rolling summary,
chat messages for UI restore, and small workspace metadata. Refreshing the
browser reloads this file; clearing the index or memory is explicit.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from memory_search import hashed_embedding, rank_memories


def _default_db_path() -> Path:
    configured = os.getenv("WORKSPACE_DB_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    data_dir = Path(
        os.getenv(
            "WORKSPACE_DATA_DIR",
            str(Path(__file__).resolve().parent / "data"),
        )
    ).expanduser()
    return data_dir / "workspace.duckdb"


class WorkspaceStore:
    """Process-wide DuckDB workspace. Safe for the single-user UI."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or _default_db_path()).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection: Any | None = None

    def _connect(self) -> Any:
        if self._connection is not None:
            return self._connection
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "DuckDB is required for workspace memory. Install duckdb."
            ) from exc
        self._connection = duckdb.connect(str(self.path))
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id VARCHAR PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                question_emb FLOAT[],
                answer_emb FLOAT[],
                created_at TIMESTAMP NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                role VARCHAR NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS workspace_meta (
                key VARCHAR PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS rolling_summary (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
            """
        )
        return self._connection

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def set_meta(self, key: str, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False, default=str)
        with self._lock:
            db = self._connect()
            db.execute(
                """
                INSERT INTO workspace_meta(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                [key, payload],
            )

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self._lock:
            db = self._connect()
            row = db.execute(
                "SELECT value FROM workspace_meta WHERE key = ?", [key]
            ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return row[0]

    def meta_snapshot(self) -> dict[str, Any]:
        with self._lock:
            db = self._connect()
            rows = db.execute("SELECT key, value FROM workspace_meta").fetchall()
        snapshot: dict[str, Any] = {}
        for key, value in rows:
            try:
                snapshot[str(key)] = json.loads(value)
            except json.JSONDecodeError:
                snapshot[str(key)] = value
        return snapshot

    def add_turn(
        self,
        question: str,
        answer: str,
        *,
        question_emb: list[float] | None = None,
        answer_emb: list[float] | None = None,
    ) -> str:
        memory_id = str(uuid4())
        q_vec = question_emb or hashed_embedding(question)
        a_vec = answer_emb or hashed_embedding(answer)
        now = datetime.now(timezone.utc)
        with self._lock:
            db = self._connect()
            db.execute(
                """
                INSERT INTO memories(
                    id, question, answer, question_emb, answer_emb, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [memory_id, question, answer, q_vec, a_vec, now],
            )
            next_ids = db.execute(
                "SELECT COALESCE(MAX(id), 0) + 1 FROM messages"
            ).fetchone()
            next_id = int(next_ids[0]) if next_ids else 1
            db.execute(
                "INSERT INTO messages(id, role, content, created_at) VALUES (?, ?, ?, ?)",
                [next_id, "user", question, now],
            )
            db.execute(
                "INSERT INTO messages(id, role, content, created_at) VALUES (?, ?, ?, ?)",
                [next_id + 1, "assistant", answer, now],
            )
        return memory_id

    def list_messages(self) -> list[dict[str, str]]:
        with self._lock:
            db = self._connect()
            rows = db.execute(
                "SELECT role, content FROM messages ORDER BY id"
            ).fetchall()
        return [{"role": str(role), "content": str(content)} for role, content in rows]

    def replace_messages(self, messages: Iterable[dict[str, str]]) -> None:
        rows = [
            (str(item.get("role") or "user"), str(item.get("content") or ""))
            for item in messages
            if str(item.get("content") or "").strip()
        ]
        now = datetime.now(timezone.utc)
        with self._lock:
            db = self._connect()
            db.execute("DELETE FROM messages")
            for index, (role, content) in enumerate(rows, start=1):
                db.execute(
                    "INSERT INTO messages(id, role, content, created_at) VALUES (?, ?, ?, ?)",
                    [index, role, content, now],
                )

    def list_memories(self) -> list[dict[str, Any]]:
        with self._lock:
            db = self._connect()
            rows = db.execute(
                """
                SELECT id, question, answer, question_emb, answer_emb, created_at
                FROM memories
                ORDER BY created_at
                """
            ).fetchall()
        results: list[dict[str, Any]] = []
        for memory_id, question, answer, question_emb, answer_emb, created_at in rows:
            results.append(
                {
                    "id": str(memory_id),
                    "question": str(question),
                    "answer": str(answer),
                    "question_emb": list(question_emb or []),
                    "answer_emb": list(answer_emb or []),
                    "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at),
                }
            )
        return results

    def search(
        self,
        query: str,
        query_vector: list[float] | None,
        *,
        threshold: float,
        limit: int,
    ) -> list[Any]:
        rows = self.list_memories()
        return rank_memories(
            query,
            query_vector or hashed_embedding(query),
            rows,
            threshold=threshold,
            limit=limit,
        )

    def get_summary(self) -> str:
        with self._lock:
            db = self._connect()
            row = db.execute(
                "SELECT content FROM rolling_summary WHERE id = 1"
            ).fetchone()
        return str(row[0]) if row else ""

    def set_summary(self, content: str) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            db = self._connect()
            db.execute(
                """
                INSERT INTO rolling_summary(id, content, updated_at) VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET content = excluded.content, updated_at = excluded.updated_at
                """,
                [content, now],
            )

    def clear_memories(self) -> None:
        with self._lock:
            db = self._connect()
            db.execute("DELETE FROM memories")
            db.execute("DELETE FROM messages")
            db.execute("DELETE FROM rolling_summary")

    def clear_workspace(self) -> None:
        with self._lock:
            db = self._connect()
            db.execute("DELETE FROM memories")
            db.execute("DELETE FROM messages")
            db.execute("DELETE FROM rolling_summary")
            db.execute("DELETE FROM workspace_meta")


STORE = WorkspaceStore()
