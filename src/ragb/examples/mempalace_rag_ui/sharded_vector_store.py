"""LangChain-compatible adapter over independently searchable SQLite shards."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import tempfile
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from sharding import (
    ProgressiveConfig,
    ProgressiveShardRetriever,
    ShardMetadata,
    ShardProfile,
    ShardQuery,
    ShardRecord,
    ShardRegistry,
    ShardRouter,
    utc_now,
)
from sqlite_vector_store import SQLiteVecStore


_SAFE_ID = re.compile(r"[^a-zA-Z0-9_.-]+")


def _bool_setting(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _dimensions() -> tuple[str, ...]:
    raw = os.getenv("RAG_SHARD_DIMENSIONS", "").strip()
    return tuple(item.strip().lower() for item in raw.split(",") if item.strip())


def _safe_shard_id(value: str) -> str:
    value = _SAFE_ID.sub("-", value.strip()).strip(".-") or "default"
    return value[:100]


def _embedding_model_version() -> str:
    return os.getenv("NVIDIA_EMBEDDING_MODEL", "embedding-provider-default")


class _SQLiteShardBackend:
    """Minimal backend protocol bridge; SQLite remains independently replaceable."""

    def __init__(self, store: SQLiteVecStore) -> None:
        self.store = store

    def search(
        self, query: str, *, k: int, filters: dict[str, Any] | None = None
    ) -> list[tuple[Document, float]]:
        return self.store.similarity_search_with_score(query, k=k, filter=filters)


class ShardedVectorStore(VectorStore):
    """A replaceable vector-store facade that routes before searching.

    With no ``RAG_SHARD_DIMENSIONS`` configured, this intentionally creates
    one ``default`` shard around the existing SQLite index. When dimensions
    are configured, documents are independently written to one SQLite file per
    partition and searched in parallel with global fusion.
    """

    def __init__(self, base_path: str | Path, embedding: Embeddings) -> None:
        self.base_path = Path(base_path).expanduser()
        self.embedding = embedding
        self.dimensions = _dimensions()
        self.enabled = _bool_setting("RAG_SHARDING_ENABLED", bool(self.dimensions))
        self.registry = ShardRegistry()
        self.router = ShardRouter(self.registry, cache_ttl_seconds=float(os.getenv("RAG_SHARD_ROUTE_CACHE_TTL", "30")))
        self.retriever = ProgressiveShardRetriever(
            self.registry,
            self.router,
            config=ProgressiveConfig(
                initial_fanout=max(1, int(os.getenv("RAG_SHARD_INITIAL_FANOUT", "2"))),
                max_fanout=max(1, int(os.getenv("RAG_SHARD_MAX_FANOUT", "8"))),
                per_shard_k=max(1, int(os.getenv("RAG_SHARD_PER_SHARD_K", "8"))),
                result_limit=max(1, int(os.getenv("RAG_SHARD_RESULT_LIMIT", "8"))),
                max_workers=max(1, int(os.getenv("RAG_SHARD_MAX_WORKERS", "4"))),
                min_results=max(1, int(os.getenv("RAG_SHARD_MIN_RESULTS", "3"))),
            ),
        )
        self._backends: dict[str, SQLiteVecStore] = {}
        # ContextVars survive Quivr's async-to-thread bridge, unlike thread
        # locals.  That keeps the ACL-filtered routing scope attached to the
        # actual vector search rather than merely the request handler.
        self._context: ContextVar[ShardQuery | None] = ContextVar(
            "shard_query", default=None
        )
        self._last_report: dict[str, Any] | None = None
        self._registry_path = self.base_path.with_suffix(".shards.json")
        self._load_registry()
        # Preserve a pre-sharding SQLite index as one legacy shard. Fresh
        # sharded indexes intentionally create no empty catch-all shard.
        if not self.registry.records() and (not self.enabled or self.base_path.exists()):
            self._ensure_backend("default")

    @property
    def last_report(self) -> dict[str, Any] | None:
        return self._last_report

    @staticmethod
    def can_restore(base_path: str | Path) -> bool:
        path = Path(base_path).expanduser()
        return path.exists() or path.with_suffix(".shards.json").exists()

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
    ) -> "ShardedVectorStore":
        """Create a standalone sharded store using the normal LangChain API."""

        del kwargs
        store = cls(
            path or (Path(tempfile.gettempdir()) / f"quivr-shards-{uuid4().hex}.sqlite3"),
            embedding=embedding,
        )
        store.add_texts(texts, metadatas, ids=ids)
        return store

    def _backend_path(self, shard_id: str) -> Path:
        if shard_id == "default" and (
            not self.enabled
            or (self.base_path.exists() and not self._registry_path.exists())
        ):
            return self.base_path
        return self.base_path.with_name(f"{self.base_path.stem}--{_safe_shard_id(shard_id)}{self.base_path.suffix}")

    def _ensure_backend(self, shard_id: str, metadata: ShardMetadata | None = None) -> SQLiteVecStore:
        if shard_id in self._backends:
            return self._backends[shard_id]
        backend_path = self._backend_path(shard_id)
        backend = SQLiteVecStore(backend_path, embedding=self.embedding)
        self._backends[shard_id] = backend
        record = ShardRecord(
            metadata=metadata or ShardMetadata(
                shard_id=shard_id,
                tenant=os.getenv("RAG_TENANT_ID", "default"),
                model_version=_embedding_model_version(),
                index_config={"dimensions": list(self.dimensions), "score_mode": "distance"},
            ),
            backend=_SQLiteShardBackend(backend),
            backend_path=str(backend_path),
        )
        self.registry.register(record)
        return backend

    def _load_registry(self) -> None:
        if not self._registry_path.is_file():
            return
        try:
            payload = json.loads(self._registry_path.read_text(encoding="utf-8"))
            for item in payload.get("shards", []):
                metadata = ShardMetadata.from_dict(item)
                if metadata.model_version != _embedding_model_version():
                    # A centroid created by another embedding model must never
                    # influence semantic routing under this model.
                    metadata.model_version = _embedding_model_version()
                path = Path(item.get("backend_path") or self._backend_path(metadata.shard_id))
                if not path.exists():
                    continue
                backend = SQLiteVecStore(path, embedding=self.embedding)
                self._backends[metadata.shard_id] = backend
                self.registry.register(
                    ShardRecord(
                        metadata=metadata,
                        backend=_SQLiteShardBackend(backend),
                        profile=(
                            ShardProfile.from_dict(item.get("profile") or {})
                            if str(item.get("model_version") or "") == _embedding_model_version()
                            else ShardProfile()
                        ),
                        backend_path=str(path),
                    )
                )
        except Exception:
            # A corrupt optional registry must not prevent the main app from
            # starting; the normal index path will rebuild it.
            return

    def _persist_registry(self) -> None:
        payload = self.registry.snapshot()
        for item in payload["shards"]:
            record = self.registry.get(item["shard_id"])
            item["backend_path"] = record.backend_path if record else ""
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._registry_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self._registry_path)

    def _shard_key(self, document: Document) -> str:
        if not self.enabled or not self.dimensions:
            return "default"
        metadata = document.metadata or {}
        values: list[str] = []
        for dimension in self.dimensions:
            if dimension == "time":
                value = metadata.get("timestamp") or metadata.get("date") or metadata.get("created_at") or "unknown"
                value = str(value)[:7]
            elif dimension == "project":
                value = metadata.get("project") or metadata.get("collection") or "default"
            else:
                value = metadata.get(dimension) or "default"
            values.append(f"{dimension}={value}")
        key = "|".join(values)
        shard_id = _safe_shard_id("--".join(values))
        max_shards = max(1, int(os.getenv("RAG_MAX_SHARDS", "32")))
        existing = {record.metadata.shard_id for record in self.registry.records()}
        if shard_id not in existing and len(existing) >= max_shards:
            bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % max_shards
            return f"hash-{bucket:02d}"
        return shard_id

    def _metadata_for(self, shard_id: str, document: Document) -> ShardMetadata:
        values = document.metadata or {}
        metadata_values = {
            key: str(value)
            for key, value in values.items()
            if key in {"tenant", "domain", "project", "collection", "region"}
        }
        return ShardMetadata(
            shard_id=shard_id,
            tenant=str(values.get("tenant") or os.getenv("RAG_TENANT_ID", "default")),
            domain=str(values.get("domain") or ""),
            project=str(values.get("project") or values.get("collection") or ""),
            region=str(values.get("region") or ""),
            time_start=str(values.get("timestamp") or values.get("date") or values.get("created_at") or "")[:10],
            time_end=str(values.get("timestamp") or values.get("date") or values.get("created_at") or "")[:10],
            index_config={
                "dimensions": list(self.dimensions),
                "score_mode": "distance",
                "metadata": metadata_values,
            },
            model_version=_embedding_model_version(),
            config_version=os.getenv("RAG_SHARD_CONFIG_VERSION", "1"),
        )

    def _profile_documents(self, documents: Iterable[Document]) -> ShardProfile:
        document_list = list(documents)
        keywords: set[str] = set()
        entities: set[str] = set()
        distribution: dict[str, dict[str, int]] = {}
        for document in document_list:
            text = str(document.page_content or "")
            words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text.lower())
            keywords.update(words[:50])
            entities.update(
                match.lower()
                for match in re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", text)[:25]
            )
            for key, value in (document.metadata or {}).items():
                bucket = distribution.setdefault(str(key), {})
                bucket[str(value)] = bucket.get(str(value), 0) + 1
        centroid: tuple[float, ...] = ()
        sample_size = max(0, int(os.getenv("RAG_SHARD_PROFILE_SIZE", "8")))
        if sample_size:
            try:
                vectors = self.embedding.embed_documents(
                    [str(document.page_content or "") for document in document_list[:sample_size]]
                )
                if vectors and vectors[0]:
                    width = len(vectors[0])
                    centroid = tuple(
                        sum(float(vector[index]) for vector in vectors) / len(vectors)
                        for index in range(width)
                    )
            except Exception:
                # Profiles are a routing optimization. A profile refresh must
                # never make independently durable document ingestion fail.
                centroid = ()
        return ShardProfile(
            centroid=centroid,
            keywords=keywords,
            entities=entities,
            metadata_distribution=distribution,
        )

    @staticmethod
    def _merge_profiles(existing: ShardProfile, incoming: ShardProfile) -> ShardProfile:
        distribution = {
            key: dict(values) for key, values in existing.metadata_distribution.items()
        }
        for key, values in incoming.metadata_distribution.items():
            target = distribution.setdefault(key, {})
            for value, count in values.items():
                target[value] = target.get(value, 0) + count
        return ShardProfile(
            centroid=incoming.centroid or existing.centroid,
            keywords=existing.keywords | incoming.keywords,
            entities=existing.entities | incoming.entities,
            metadata_distribution=distribution,
        )

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        if not documents:
            return []
        requested_ids = kwargs.pop("ids", None)
        if requested_ids is not None and len(requested_ids) != len(documents):
            raise ValueError("The number of ids must match the number of documents.")
        groups: dict[str, list[tuple[int, Document]]] = {}
        for index, document in enumerate(documents):
            groups.setdefault(self._shard_key(document), []).append((index, document))
        ids: list[str] = []
        for shard_id, indexed_group in groups.items():
            group = [document for _, document in indexed_group]
            backend = self._ensure_backend(shard_id, self._metadata_for(shard_id, group[0]))
            group_kwargs = dict(kwargs)
            if requested_ids is not None:
                group_kwargs["ids"] = [requested_ids[index] for index, _ in indexed_group]
            ids.extend(backend.add_documents(group, **group_kwargs))
            record = self.registry.get(shard_id)
            if record:
                record.metadata.chunk_count += len(group)
                document_sources = {
                    str(document.metadata.get("source") or document.id or "")
                    for document in group
                }
                record.metadata.document_count += len(document_sources)
                record.metadata.updated_at = utc_now()
                self.registry.update_profile(
                    shard_id,
                    self._merge_profiles(record.profile, self._profile_documents(group)),
                )
        self._persist_registry()
        return ids

    async def aadd_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        return await asyncio.to_thread(self.add_documents, documents, **kwargs)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        values = list(texts)
        metadata_values = metadatas or [{} for _ in values]
        if len(metadata_values) != len(values):
            raise ValueError("The number of metadatas must match the number of texts.")
        if ids is not None and len(ids) != len(values):
            raise ValueError("The number of ids must match the number of texts.")
        documents = [
            Document(id=ids[index] if ids else None, page_content=text, metadata=metadata)
            for index, (text, metadata) in enumerate(zip(values, metadata_values, strict=True))
        ]
        return self.add_documents(documents, ids=ids, **kwargs)

    @contextmanager
    def query_context(self, context: ShardQuery):
        token = self._context.set(context)
        try:
            yield
        finally:
            self._context.reset(token)

    def _current_context(self) -> ShardQuery:
        return self._context.get() or ShardQuery(
            tenant=os.getenv("RAG_TENANT_ID", "default"),
            security_scope=os.getenv("RAG_SECURITY_SCOPE", "default"),
            authorized_shards=frozenset(
                item.strip()
                for item in os.getenv("RAG_ALLOWED_SHARDS", "").split(",")
                if item.strip()
            )
            or None,
        )

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs: Any) -> list[tuple[Document, float]]:
        context = kwargs.pop("shard_query", None) or self._current_context()
        if not context.embedding and any(
            record.profile.centroid for record in self.registry.records()
        ):
            try:
                context = replace(context, embedding=tuple(self.embedding.embed_query(query)))
            except Exception:
                # Keyword/metadata routing remains available if the profile
                # model is temporarily unavailable.
                pass
        report = self.retriever.search(query, context=context, filters=kwargs.pop("filter", None))
        self._last_report = report.snapshot()
        return [(item.document, max(0.0, 1.0 - item.fused_score)) for item in report.results[:k]]

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        return [document for document, _ in self.similarity_search_with_score(query, k=k, **kwargs)]

    async def asimilarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        return await asyncio.to_thread(self.similarity_search_with_score, query, k, **kwargs)

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        return await asyncio.to_thread(self.similarity_search, query, k, **kwargs)

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool:
        del kwargs
        for backend in self._backends.values():
            backend.delete(ids)
        return True

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "dimensions": list(self.dimensions),
            "registry": self.registry.snapshot(),
            "last_retrieval": self._last_report,
            "lifecycle_recommendations": self.lifecycle_recommendations(),
        }

    def set_shard_status(self, shard_id: str, status: str) -> dict[str, Any]:
        """Administrative lifecycle control; routing excludes unavailable states."""

        record = self.registry.update_metadata(shard_id, status=status)
        self._persist_registry()
        return record.metadata.to_dict()

    def lifecycle_recommendations(self) -> list[dict[str, Any]]:
        """Return safe split/merge recommendations for an index migration job.

        Data movement is intentionally outside the router: the migration job
        must preserve document mappings, provenance, ACL boundaries, and index
        compatibility. Marking a shard ``migrating`` or ``rebuilding`` removes
        it from routing until that job completes.
        """

        split_at = max(1, int(os.getenv("RAG_SHARD_SPLIT_CHUNKS", "50000")))
        merge_below = max(0, int(os.getenv("RAG_SHARD_MERGE_CHUNKS", "1000")))
        records = self.registry.records()
        actions: list[dict[str, Any]] = []
        for record in records:
            metadata = record.metadata
            if metadata.chunk_count >= split_at:
                actions.append(
                    {
                        "action": "split",
                        "shard_id": metadata.shard_id,
                        "reason": f"chunk_count {metadata.chunk_count} >= {split_at}",
                    }
                )
        small = [
            record.metadata
            for record in records
            if 0 < record.metadata.chunk_count <= merge_below
            and record.metadata.status == "healthy"
        ]
        for index, first in enumerate(small):
            for second in small[index + 1 :]:
                if (
                    first.tenant == second.tenant
                    and first.domain == second.domain
                    and first.project == second.project
                ):
                    actions.append(
                        {
                            "action": "merge",
                            "shard_ids": [first.shard_id, second.shard_id],
                            "reason": f"both are at or below {merge_below} chunks in the same ACL scope",
                        }
                    )
        return actions

    def close(self) -> None:
        for backend in self._backends.values():
            backend.close()

    def cleanup(self) -> None:
        paths = [Path(record.backend_path) for record in self.registry.records() if record.backend_path]
        self.close()
        for path in paths:
            for suffix in ("", "-wal", "-shm"):
                path.with_name(path.name + suffix).unlink(missing_ok=True)
        self._registry_path.unlink(missing_ok=True)


__all__ = ["ShardedVectorStore"]
