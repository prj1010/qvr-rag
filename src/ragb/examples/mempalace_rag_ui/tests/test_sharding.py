"""Unit tests for framework-independent shard routing and retrieval."""

from __future__ import annotations

import os
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from sharding import (
    ProgressiveConfig,
    ProgressiveShardRetriever,
    ShardMetadata,
    ShardQuery,
    ShardRecord,
    ShardRegistry,
    ShardRouter,
)
from sharded_vector_store import ShardedVectorStore


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict[str, str] = field(default_factory=dict)
    id: str | None = None


class FakeBackend:
    def __init__(self, rows: list[tuple[FakeDocument, float]] | None = None, *, fail: bool = False) -> None:
        self.rows = rows or []
        self.fail = fail
        self.calls = 0

    def search(self, query: str, *, k: int, filters=None):
        del query, filters
        self.calls += 1
        if self.fail:
            raise RuntimeError("backend unavailable")
        return self.rows[:k]


class TinyEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        value = text.lower()
        return [float("finance" in value), float("engineering" in value), 1.0]


def record(shard_id: str, backend: FakeBackend, **metadata: str) -> ShardRecord:
    return ShardRecord(metadata=ShardMetadata(shard_id=shard_id, **metadata), backend=backend)


class ShardRoutingTests(unittest.TestCase):
    def test_acl_is_applied_before_backend_selection(self) -> None:
        registry = ShardRegistry()
        allowed = FakeBackend([(FakeDocument("allowed", id="a"), 0.1)])
        denied = FakeBackend([(FakeDocument("secret", id="b"), 0.0)])
        registry.register(record("tenant-a", allowed, tenant="a"))
        registry.register(record("tenant-b", denied, tenant="b"))
        router = ShardRouter(registry)

        decision = router.route(
            "secret",
            context=ShardQuery.from_values(authorized_shards={"tenant-a"}),
            limit=3,
        )

        self.assertEqual([item.metadata.shard_id for item in decision.candidates], ["tenant-a"])
        self.assertEqual(decision.denied_shards, ["tenant-b"])
        self.assertEqual(denied.calls, 0)

    def test_progressive_expansion_fuses_and_deduplicates(self) -> None:
        registry = ShardRegistry()
        first = FakeBackend([(FakeDocument("one", {"source": "first"}, "same"), 0.2)])
        second = FakeBackend(
            [
                (FakeDocument("duplicate", {"source": "second"}, "same"), 0.1),
                (FakeDocument("two", {"source": "second"}, "two"), 0.3),
            ]
        )
        registry.register(record("first", first))
        registry.register(record("second", second))
        retriever = ProgressiveShardRetriever(
            registry,
            ShardRouter(registry),
            config=ProgressiveConfig(
                initial_fanout=1,
                max_fanout=2,
                per_shard_k=3,
                result_limit=3,
                min_results=2,
                min_score_gap=0.0,
            ),
        )

        report = retriever.search("question")

        self.assertTrue(report.expanded)
        self.assertEqual(report.selected_shards, ["first", "second"])
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 1)
        self.assertEqual(len(report.results), 2)
        self.assertEqual({item.document.id for item in report.results}, {"same", "two"})

    def test_failure_degrades_shard_and_invalidates_routing_cache(self) -> None:
        registry = ShardRegistry()
        backend = FakeBackend(fail=True)
        registry.register(record("unhealthy", backend))
        router = ShardRouter(registry, cache_ttl_seconds=60)
        retriever = ProgressiveShardRetriever(
            registry,
            router,
            config=ProgressiveConfig(initial_fanout=1, max_fanout=1),
        )

        report = retriever.search("question")

        self.assertIn("unhealthy", report.failures)
        self.assertEqual(registry.get("unhealthy").metadata.status, "degraded")
        cached = router.route("question")
        self.assertFalse(cached.cache_hit)

    def test_langchain_adapter_partitions_and_enforces_allow_list(self) -> None:
        previous_enabled = os.environ.get("RAG_SHARDING_ENABLED")
        previous_dimensions = os.environ.get("RAG_SHARD_DIMENSIONS")
        store = None
        try:
            os.environ["RAG_SHARDING_ENABLED"] = "true"
            os.environ["RAG_SHARD_DIMENSIONS"] = "domain"
            path = Path.cwd() / f"test-shards-{uuid4().hex}.sqlite3"
            store = ShardedVectorStore(path, TinyEmbeddings())
            store.add_documents(
                [
                    Document(page_content="finance report", metadata={"domain": "finance"}),
                    Document(page_content="engineering plan", metadata={"domain": "engineering"}),
                ]
            )
            shard_ids = {item["shard_id"] for item in store.snapshot()["registry"]["shards"]}
            finance_shard = next(item for item in shard_ids if "finance" in item)
            with store.query_context(ShardQuery.from_values(authorized_shards={finance_shard})):
                results = store.similarity_search("engineering", k=3)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].metadata["domain"], "finance")
        finally:
            if store is not None:
                store.cleanup()
            if previous_enabled is None:
                os.environ.pop("RAG_SHARDING_ENABLED", None)
            else:
                os.environ["RAG_SHARDING_ENABLED"] = previous_enabled
            if previous_dimensions is None:
                os.environ.pop("RAG_SHARD_DIMENSIONS", None)
            else:
                os.environ["RAG_SHARD_DIMENSIONS"] = previous_dimensions


if __name__ == "__main__":
    unittest.main()
