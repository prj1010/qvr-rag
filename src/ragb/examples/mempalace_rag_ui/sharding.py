"""Framework-agnostic shard routing, progressive retrieval, and fusion.

The module deliberately knows nothing about Quivr, LangChain, ingestion, or
the LLM.  A shard is a searchable backend plus a registry record; callers can
replace either side independently.  The LangChain adapter lives in
``sharded_vector_store.py``.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Protocol, Sequence


HEALTHY_STATES = frozenset({"healthy", "degraded"})
VALID_STATES = frozenset({"healthy", "degraded", "offline", "rebuilding", "migrating"})
TOKEN_RE = re.compile(r"[\w-]{2,}", re.UNICODE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tokenize(value: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(value or "")}


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(math.fsum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(math.fsum(float(value) ** 2 for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return math.fsum(float(a) * float(b) for a, b in zip(left, right)) / (
        left_norm * right_norm
    )


class SearchBackend(Protocol):
    def search(
        self, query: str, *, k: int, filters: dict[str, Any] | None = None
    ) -> list[tuple[Any, float]]: ...


@dataclass
class ShardMetadata:
    shard_id: str
    tenant: str = "default"
    domain: str = ""
    project: str = ""
    region: str = ""
    time_start: str = ""
    time_end: str = ""
    document_count: int = 0
    chunk_count: int = 0
    index_config: dict[str, Any] = field(default_factory=dict)
    index_version: str = "1"
    config_version: str = "1"
    model_version: str = ""
    status: str = "healthy"
    health_score: float = 1.0
    active_queries: int = 0
    last_error: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if self.status not in VALID_STATES:
            raise ValueError(f"Unknown shard state: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "tenant": self.tenant,
            "domain": self.domain,
            "project": self.project,
            "region": self.region,
            "time_range": {"start": self.time_start, "end": self.time_end},
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "index_config": dict(self.index_config),
            "index_version": self.index_version,
            "config_version": self.config_version,
            "model_version": self.model_version,
            "status": self.status,
            "health_score": self.health_score,
            "active_queries": self.active_queries,
            "last_error": self.last_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ShardMetadata":
        time_range = value.get("time_range") or {}
        return cls(
            shard_id=str(value["shard_id"]),
            tenant=str(value.get("tenant", "default")),
            domain=str(value.get("domain", "")),
            project=str(value.get("project", "")),
            region=str(value.get("region", "")),
            time_start=str(value.get("time_start", time_range.get("start", ""))),
            time_end=str(value.get("time_end", time_range.get("end", ""))),
            document_count=int(value.get("document_count", 0)),
            chunk_count=int(value.get("chunk_count", 0)),
            index_config=dict(value.get("index_config") or {}),
            index_version=str(value.get("index_version", "1")),
            config_version=str(value.get("config_version", "1")),
            model_version=str(value.get("model_version", "")),
            status=str(value.get("status", "healthy")),
            health_score=float(value.get("health_score", 1.0)),
            active_queries=int(value.get("active_queries", 0)),
            last_error=str(value.get("last_error", "")),
            created_at=str(value.get("created_at", utc_now())),
            updated_at=str(value.get("updated_at", utc_now())),
        )


@dataclass
class ShardProfile:
    centroid: tuple[float, ...] = ()
    keywords: set[str] = field(default_factory=set)
    entities: set[str] = field(default_factory=set)
    metadata_distribution: dict[str, dict[str, int]] = field(default_factory=dict)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "centroid": list(self.centroid),
            "keywords": sorted(self.keywords),
            "entities": sorted(self.entities),
            "metadata_distribution": self.metadata_distribution,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ShardProfile":
        return cls(
            centroid=tuple(float(item) for item in value.get("centroid") or ()),
            keywords=set(str(item) for item in value.get("keywords") or ()),
            entities=set(str(item) for item in value.get("entities") or ()),
            metadata_distribution={
                str(key): {str(inner_key): int(count) for inner_key, count in values.items()}
                for key, values in (value.get("metadata_distribution") or {}).items()
                if isinstance(values, dict)
            },
            updated_at=str(value.get("updated_at") or utc_now()),
        )


@dataclass
class ShardRecord:
    metadata: ShardMetadata
    backend: SearchBackend
    profile: ShardProfile = field(default_factory=ShardProfile)
    backend_path: str = ""


@dataclass(frozen=True)
class ShardQuery:
    security_scope: str = "default"
    authorized_shards: frozenset[str] | None = None
    tenant: str = ""
    domain: str = ""
    project: str = ""
    time_start: str = ""
    time_end: str = ""
    filters: tuple[tuple[str, str], ...] = ()
    keywords: frozenset[str] = frozenset()
    entities: frozenset[str] = frozenset()
    embedding: tuple[float, ...] = ()
    historical_shards: frozenset[str] = frozenset()

    @classmethod
    def from_values(cls, **values: Any) -> "ShardQuery":
        filters = values.get("filters") or {}
        if isinstance(filters, dict):
            values["filters"] = tuple(sorted((str(k), str(v)) for k, v in filters.items()))
        for key in ("keywords", "entities"):
            values[key] = frozenset(values.get(key) or ())
        if values.get("authorized_shards") is not None:
            values["authorized_shards"] = frozenset(values["authorized_shards"])
        if values.get("embedding") is not None:
            values["embedding"] = tuple(float(item) for item in values["embedding"])
        return cls(**values)

    @property
    def filter_dict(self) -> dict[str, str]:
        return dict(self.filters)


@dataclass(frozen=True)
class RoutingWeights:
    semantic: float = 0.40
    metadata: float = 0.22
    keyword: float = 0.14
    entity: float = 0.10
    temporal: float = 0.06
    freshness: float = 0.04
    historical: float = 0.02
    health: float = 0.02


@dataclass
class RoutingDecision:
    candidates: list[ShardRecord]
    authorized_count: int
    denied_shards: list[str] = field(default_factory=list)
    cache_hit: bool = False
    router_version: int = 0


class ShardRegistry:
    """Thread-safe registry with versioned metadata and profile invalidation."""

    def __init__(self) -> None:
        self._records: dict[str, ShardRecord] = {}
        self._version = 0
        self._lock = threading.RLock()

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def register(self, record: ShardRecord) -> None:
        with self._lock:
            self._records[record.metadata.shard_id] = record
            self._version += 1

    def remove(self, shard_id: str) -> ShardRecord | None:
        with self._lock:
            removed = self._records.pop(shard_id, None)
            if removed is not None:
                self._version += 1
            return removed

    def get(self, shard_id: str) -> ShardRecord | None:
        with self._lock:
            return self._records.get(shard_id)

    def records(self) -> list[ShardRecord]:
        with self._lock:
            return list(self._records.values())

    def update_metadata(self, shard_id: str, **changes: Any) -> ShardRecord:
        with self._lock:
            record = self._records[shard_id]
            if "status" in changes and changes["status"] not in VALID_STATES:
                raise ValueError(f"Unknown shard state: {changes['status']}")
            record.metadata = replace(record.metadata, **changes, updated_at=utc_now())
            self._version += 1
            return record

    def update_profile(self, shard_id: str, profile: ShardProfile) -> None:
        with self._lock:
            self._records[shard_id].profile = profile
            self._version += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": self._version,
                "shards": [
                    {
                        **record.metadata.to_dict(),
                        "profile": record.profile.to_dict(),
                    }
                    for record in self._records.values()
                ],
            }


class ShardRouter:
    """ACL-first deterministic router with versioned, scope-aware caching."""

    def __init__(
        self,
        registry: ShardRegistry,
        *,
        weights: RoutingWeights | None = None,
        cache_ttl_seconds: float = 30.0,
    ) -> None:
        self.registry = registry
        self.weights = weights or RoutingWeights()
        self.cache_ttl_seconds = max(0.0, cache_ttl_seconds)
        self._cache: dict[tuple[Any, ...], tuple[float, RoutingDecision]] = {}
        self._lock = threading.RLock()

    def _cache_key(self, query: str, context: ShardQuery, limit: int) -> tuple[Any, ...]:
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
        shard_versions = tuple(
            sorted(
                (
                    record.metadata.shard_id,
                    record.metadata.index_version,
                    record.metadata.config_version,
                    record.metadata.model_version,
                )
                for record in self.registry.records()
            )
        )
        return (
            self.registry.version,
            shard_versions,
            context.security_scope,
            digest,
            context.tenant,
            context.domain,
            context.project,
            context.time_start,
            context.time_end,
            context.filters,
            tuple(sorted(context.keywords)),
            tuple(sorted(context.entities)),
            tuple(sorted(context.authorized_shards or ("*",))),
            tuple(sorted(context.historical_shards)),
            limit,
        )

    @staticmethod
    def _explicit_match(metadata: ShardMetadata, context: ShardQuery) -> bool:
        for requested, actual in (
            (context.tenant, metadata.tenant),
            (context.domain, metadata.domain),
            (context.project, metadata.project),
        ):
            if requested and actual and actual not in {requested, "*"}:
                return False
        if context.time_start and metadata.time_end and metadata.time_end < context.time_start:
            return False
        if context.time_end and metadata.time_start and metadata.time_start > context.time_end:
            return False
        for key, value in context.filters:
            actual = metadata.index_config.get("metadata", {}).get(key)
            if actual is not None and str(actual) != value:
                return False
        return True

    def _score(self, query: str, record: ShardRecord, context: ShardQuery) -> float:
        metadata = record.metadata
        profile = record.profile
        query_terms = tokenize(query) | set(context.keywords)
        keyword_score = len(query_terms & profile.keywords) / max(1, len(query_terms))
        entity_score = len(set(context.entities) & profile.entities) / max(1, len(context.entities))
        semantic_score = cosine_similarity(context.embedding, profile.centroid)
        metadata_score = sum(
            bool(requested and actual and requested == actual)
            for requested, actual in (
                (context.tenant, metadata.tenant),
                (context.domain, metadata.domain),
                (context.project, metadata.project),
            )
        ) / 3.0
        temporal_score = 1.0 if context.time_start or context.time_end else 0.0
        historical_score = 1.0 if metadata.shard_id in context.historical_shards else 0.0
        freshness_score = 1.0 if metadata.status == "healthy" else 0.5
        load_score = 1.0 / (1.0 + max(0, metadata.active_queries))
        w = self.weights
        return (
            w.semantic * semantic_score
            + w.metadata * metadata_score
            + w.keyword * keyword_score
            + w.entity * entity_score
            + w.temporal * temporal_score
            + w.freshness * freshness_score
            + w.historical * historical_score
            + w.health * max(0.0, min(1.0, metadata.health_score)) * load_score
        )

    def route(
        self,
        query: str,
        *,
        context: ShardQuery | None = None,
        limit: int = 3,
    ) -> RoutingDecision:
        context = context or ShardQuery()
        limit = max(1, limit)
        key = self._cache_key(query, context, limit)
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(key)
            if cached and now - cached[0] <= self.cache_ttl_seconds:
                decision = cached[1]
                return RoutingDecision(
                    candidates=list(decision.candidates),
                    authorized_count=decision.authorized_count,
                    denied_shards=list(decision.denied_shards),
                    cache_hit=True,
                    router_version=self.registry.version,
                )

        records = self.registry.records()
        authorized: list[ShardRecord] = []
        denied: list[str] = []
        allowed = context.authorized_shards
        for record in records:
            shard_id = record.metadata.shard_id
            # This is intentionally the first gate. No score is computed for a
            # shard that is outside the caller's authorization scope.
            if allowed is not None and shard_id not in allowed:
                denied.append(shard_id)
                continue
            if record.metadata.status not in HEALTHY_STATES:
                continue
            if not self._explicit_match(record.metadata, context):
                continue
            authorized.append(record)

        authorized.sort(key=lambda record: self._score(query, record, context), reverse=True)
        decision = RoutingDecision(
            candidates=authorized[:limit],
            authorized_count=len(authorized),
            denied_shards=denied,
            router_version=self.registry.version,
        )
        with self._lock:
            self._cache[key] = (now, decision)
            if len(self._cache) > 2048:
                self._cache.clear()
        return decision


@dataclass
class RetrievedDocument:
    document: Any
    shard_id: str
    raw_score: float
    normalized_score: float = 0.0
    fused_score: float = 0.0
    rank: int = 0


@dataclass
class RetrievalReport:
    results: list[RetrievedDocument]
    selected_shards: list[str]
    attempted_shards: list[str]
    expanded: bool
    failures: dict[str, str]
    router_cache_hit: bool
    authorized_shards: int
    latency_ms: float
    evidence: dict[str, Any]

    def snapshot(self) -> dict[str, Any]:
        return {
            "selected_shards": self.selected_shards,
            "attempted_shards": self.attempted_shards,
            "expanded": self.expanded,
            "failures": dict(self.failures),
            "router_cache_hit": self.router_cache_hit,
            "authorized_shards": self.authorized_shards,
            "latency_ms": round(self.latency_ms, 2),
            "result_count": len(self.results),
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class ProgressiveConfig:
    initial_fanout: int = 2
    max_fanout: int = 8
    per_shard_k: int = 8
    result_limit: int = 8
    max_workers: int = 4
    min_results: int = 3
    min_top_score: float = 0.22
    min_score_gap: float = 0.03
    min_source_agreement: float = 0.0
    rrf_k: int = 60
    score_mode: str = "distance"


class ProgressiveShardRetriever:
    """Parallel shard retrieval with evidence-driven expansion and global fusion."""

    def __init__(
        self,
        registry: ShardRegistry,
        router: ShardRouter,
        *,
        config: ProgressiveConfig | None = None,
        reranker: Callable[[str, list[RetrievedDocument]], list[RetrievedDocument]] | None = None,
    ) -> None:
        self.registry = registry
        self.router = router
        self.config = config or ProgressiveConfig()
        self.reranker = reranker

    @staticmethod
    def _document_key(document: Any) -> str:
        identifier = getattr(document, "id", None)
        if identifier:
            return str(identifier)
        metadata = getattr(document, "metadata", {}) or {}
        source = metadata.get("source") or metadata.get("original_file_name") or ""
        chunk = metadata.get("chunk_index") or metadata.get("chunk_id") or ""
        content = getattr(document, "page_content", None) or str(document)
        return hashlib.sha256(f"{source}|{chunk}|{content}".encode()).hexdigest()

    def _search_shard(
        self, record: ShardRecord, query: str, filters: dict[str, Any] | None
    ) -> list[RetrievedDocument]:
        record.metadata.active_queries += 1
        try:
            matches = record.backend.search(query, k=self.config.per_shard_k, filters=filters)
            return [
                RetrievedDocument(document=item, shard_id=record.metadata.shard_id, raw_score=float(score))
                for item, score in matches
            ]
        finally:
            record.metadata.active_queries = max(0, record.metadata.active_queries - 1)

    def _retrieve(
        self,
        records: list[ShardRecord],
        query: str,
        filters: dict[str, Any] | None,
    ) -> tuple[list[RetrievedDocument], dict[str, str]]:
        failures: dict[str, str] = {}
        collected: list[RetrievedDocument] = []
        workers = max(1, min(self.config.max_workers, len(records)))
        if not records:
            return collected, failures
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="shard-retrieval") as pool:
            futures = {pool.submit(self._search_shard, record, query, filters): record for record in records}
            for future in as_completed(futures):
                record = futures[future]
                try:
                    collected.extend(future.result())
                except Exception as exc:
                    shard_id = record.metadata.shard_id
                    failures[shard_id] = f"{type(exc).__name__}: {exc}"
                    self.registry.update_metadata(
                        shard_id,
                        status="degraded",
                        last_error=failures[shard_id],
                    )
        return collected, failures

    def _fuse(self, documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
        if not documents:
            return []
        by_shard: dict[str, list[RetrievedDocument]] = {}
        for item in documents:
            by_shard.setdefault(item.shard_id, []).append(item)
        for items in by_shard.values():
            items.sort(key=lambda item: item.raw_score, reverse=self.config.score_mode != "distance")
            values = [
                (1.0 - item.raw_score if self.config.score_mode == "distance" else item.raw_score)
                for item in items
            ]
            low, high = min(values), max(values)
            for rank, (item, value) in enumerate(zip(items, values, strict=True), start=1):
                item.rank = rank
                item.normalized_score = (value - low) / (high - low) if high > low else max(0.0, value)
                item.fused_score = item.normalized_score + 1.0 / (self.config.rrf_k + rank)

        deduped: dict[str, RetrievedDocument] = {}
        for item in documents:
            key = self._document_key(item.document)
            existing = deduped.get(key)
            if existing is None or item.fused_score > existing.fused_score:
                deduped[key] = item
        return sorted(deduped.values(), key=lambda item: item.fused_score, reverse=True)

    @staticmethod
    def _evidence(documents: list[RetrievedDocument]) -> dict[str, Any]:
        if not documents:
            return {"candidate_count": 0, "top_score": 0.0, "score_gap": 0.0, "source_agreement": 0.0}
        sources = {
            str((getattr(item.document, "metadata", {}) or {}).get("source") or item.shard_id)
            for item in documents
        }
        top = documents[0].fused_score
        second = documents[1].fused_score if len(documents) > 1 else 0.0
        return {
            "candidate_count": len(documents),
            "top_score": round(top, 6),
            "score_gap": round(max(0.0, top - second), 6),
            "source_agreement": round(len(sources) / max(1, len(documents)), 6),
        }

    def _needs_expansion(self, evidence: dict[str, Any]) -> bool:
        return (
            evidence["candidate_count"] < self.config.min_results
            or evidence["top_score"] < self.config.min_top_score
            or evidence["score_gap"] < self.config.min_score_gap
            or evidence["source_agreement"] < self.config.min_source_agreement
        )

    def search(
        self,
        query: str,
        *,
        context: ShardQuery | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalReport:
        started = time.perf_counter()
        context = context or ShardQuery()
        decision = self.router.route(query, context=context, limit=max(1, self.config.max_fanout))
        all_candidates = decision.candidates
        selected: list[ShardRecord] = []
        results: list[RetrievedDocument] = []
        failures: dict[str, str] = {}
        expanded = False
        cursor = 0
        fanout = min(self.config.initial_fanout, len(all_candidates))
        while cursor < len(all_candidates) and fanout:
            batch = all_candidates[cursor : fanout]
            selected.extend(record for record in batch if record not in selected)
            batch_results, batch_failures = self._retrieve(batch, query, filters or context.filter_dict)
            results.extend(batch_results)
            failures.update(batch_failures)
            fused = self._fuse(results)
            evidence = self._evidence(fused)
            cursor = fanout
            if not self._needs_expansion(evidence) or cursor >= len(all_candidates):
                break
            expanded = True
            fanout = min(len(all_candidates), max(cursor + 1, fanout * 2, self.config.initial_fanout))

        fused = self._fuse(results)
        if self.reranker:
            fused = self.reranker(query, fused)
        fused = fused[: self.config.result_limit]
        return RetrievalReport(
            results=fused,
            selected_shards=[record.metadata.shard_id for record in selected],
            attempted_shards=[record.metadata.shard_id for record in selected],
            expanded=expanded,
            failures=failures,
            router_cache_hit=decision.cache_hit,
            authorized_shards=decision.authorized_count,
            latency_ms=(time.perf_counter() - started) * 1000,
            evidence=self._evidence(fused),
        )


__all__ = [
    "ProgressiveConfig",
    "ProgressiveShardRetriever",
    "RetrievedDocument",
    "RoutingWeights",
    "SearchBackend",
    "ShardMetadata",
    "ShardProfile",
    "ShardQuery",
    "ShardRecord",
    "ShardRegistry",
    "ShardRouter",
    "RetrievalReport",
    "cosine_similarity",
]
