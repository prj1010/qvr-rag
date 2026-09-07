"""Small SQLite-backed LangChain vector store for memory-constrained deployments.

The default Quivr vector store is FAISS, which keeps the complete index in the
Python process.  This adapter stores document text, metadata, and embeddings in
SQLite using sqlite-vec so the process does not retain a second Python copy of
the index after ingestion.

This is intentionally scoped to the operations used by the MemPalace UI.  A
durable production deployment should place the database on persistent storage
or use a managed pgvector service rather than relying on Render's ephemeral
filesystem.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from threading import RLock
from typing import Any, Iterable
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class SQLiteVecStore(VectorStore):
    """A compact disk-backed cosine vector store implemented with sqlite-vec."""

    def __init__(
        self,
        path: str | Path,
        embedding: Embeddings,
    ) -> None:
        try:
            import sqlite_vec
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "sqlite-vec is required for the SQLite vector store."
            ) from exc

        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedding = embedding
        self._lock = RLock()
        self._db = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            isolation_level=None,
        )
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA temp_store=FILE")
        self._db.enable_load_extension(True)
        sqlite_vec.load(self._db)
        self._db.enable_load_extension(False)
        self._dimension: int | None = None
        self._closed = False

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        path: str | Path | None = None,
        **kwargs: Any,
    ) -> "SQLiteVecStore":
        del kwargs
        store = cls(
            path or (Path(tempfile.gettempdir()) / f"quivr-vectors-{uuid4().hex}.sqlite3"),
            embedding=embedding,
        )
        store.add_texts(texts, metadatas, ids=ids)
        return store

    def _ensure_schema(self, dimension: int) -> None:
        if self._dimension is not None:
            if self._dimension != dimension:
                raise ValueError(
                    f"Embedding dimension changed from {self._dimension} to {dimension}."
                )
            return

        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id INTEGER PRIMARY KEY,
                external_id TEXT NOT NULL UNIQUE,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        self._db.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vectors USING vec0(
                document_id INTEGER PRIMARY KEY,
                embedding float[{int(dimension)}] distance_metric=cosine
            )
            """
        )
        self._dimension = dimension

    @staticmethod
    def _metadata_json(metadata: dict[str, Any]) -> str:
        return json.dumps(metadata, ensure_ascii=False, default=str)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        del kwargs
        text_list = list(texts)
        if not text_list:
            return []
        if metadatas is not None and len(metadatas) != len(text_list):
            raise ValueError("The number of metadatas must match the number of texts.")
        if ids is not None and len(ids) != len(text_list):
            raise ValueError("The number of ids must match the number of texts.")

        metadata_list = metadatas or [{} for _ in text_list]
        external_ids = ids or [str(uuid4()) for _ in text_list]
        from sqlite_vec import serialize_float32

        try:
            embedding_batch_size = int(
                os.getenv("SQLITE_VEC_EMBED_BATCH_SIZE", "50")
            )
        except ValueError as exc:
            raise ValueError("SQLITE_VEC_EMBED_BATCH_SIZE must be an integer.") from exc
        if embedding_batch_size <= 0:
            raise ValueError("SQLITE_VEC_EMBED_BATCH_SIZE must be positive.")

        with self._lock:
            self._db.execute("BEGIN")
            try:
                result_ids: list[str] = []
                for start in range(0, len(text_list), embedding_batch_size):
                    end = min(start + embedding_batch_size, len(text_list))
                    batch_texts = text_list[start:end]
                    vectors = self.embedding.embed_documents(batch_texts)
                    if (
                        len(vectors) != len(batch_texts)
                        or not vectors
                        or not vectors[0]
                    ):
                        raise ValueError(
                            "The embedding provider returned no document vectors."
                        )
                    self._ensure_schema(len(vectors[0]))
                    for content, metadata, external_id, vector in zip(
                        batch_texts,
                        metadata_list[start:end],
                        external_ids[start:end],
                        vectors,
                        strict=True,
                    ):
                        existing = self._db.execute(
                            "SELECT document_id FROM documents WHERE external_id = ?",
                            (external_id,),
                        ).fetchone()
                        if existing is None:
                            document_id = int(
                                self._db.execute(
                                    "SELECT COALESCE(MAX(document_id), 0) + 1 FROM documents"
                                ).fetchone()[0]
                            )
                        else:
                            document_id = int(existing[0])
                            self._db.execute(
                                "DELETE FROM vectors WHERE document_id = ?",
                                (document_id,),
                            )

                        self._db.execute(
                            """
                            INSERT INTO documents(
                                document_id, external_id, content, metadata_json
                            )
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT(external_id) DO UPDATE SET
                                content = excluded.content,
                                metadata_json = excluded.metadata_json
                            """,
                            (
                                document_id,
                                external_id,
                                content,
                                self._metadata_json(metadata),
                            ),
                        )
                        self._db.execute(
                            "INSERT INTO vectors(document_id, embedding) VALUES (?, ?)",
                            (document_id, serialize_float32(vector)),
                        )
                        result_ids.append(external_id)
                self._db.execute("COMMIT")
                return result_ids
            except Exception:
                self._db.execute("ROLLBACK")
                raise

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        *,
        filter: Any = None,
        fetch_k: int | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        del filter, kwargs
        if k < 1:
            return []
        query_vector = self.embedding.embed_query(query)
        from sqlite_vec import serialize_float32

        with self._lock:
            if self._dimension is None:
                return []
            limit = max(k, int(fetch_k or k))
            matches = self._db.execute(
                """
                SELECT document_id, distance
                FROM vectors
                WHERE embedding MATCH ? AND k = ?
                ORDER BY distance
                """,
                (serialize_float32(query_vector), limit),
            ).fetchall()
            results: list[tuple[Document, float]] = []
            for document_id, distance in matches[:k]:
                row = self._db.execute(
                    """
                    SELECT external_id, content, metadata_json
                    FROM documents WHERE document_id = ?
                    """,
                    (document_id,),
                ).fetchone()
                if row is None:
                    continue
                external_id, content, metadata_json = row
                results.append(
                    (
                        Document(
                            id=external_id,
                            page_content=content,
                            metadata=json.loads(metadata_json),
                        ),
                        float(distance),
                    )
                )
            return results

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        return [
            document
            for document, _ in self.similarity_search_with_score(query, k=k, **kwargs)
        ]

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool:
        del kwargs
        with self._lock:
            if ids is None:
                self._db.execute("DELETE FROM vectors")
                self._db.execute("DELETE FROM documents")
                return True
            for external_id in ids:
                row = self._db.execute(
                    "SELECT document_id FROM documents WHERE external_id = ?",
                    (external_id,),
                ).fetchone()
                if row is not None:
                    self._db.execute("DELETE FROM vectors WHERE document_id = ?", (row[0],))
                    self._db.execute("DELETE FROM documents WHERE document_id = ?", (row[0],))
            return True

    def close(self) -> None:
        if not self._closed:
            with self._lock:
                if not self._closed:
                    self._db.close()
                    self._closed = True

    def cleanup(self) -> None:
        """Close the store and remove its ephemeral SQLite files."""
        self.close()
        for suffix in ("", "-wal", "-shm"):
            self.path.with_name(self.path.name + suffix).unlink(missing_ok=True)
