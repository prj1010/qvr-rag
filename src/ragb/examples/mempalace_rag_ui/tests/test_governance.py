import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import governance


class GovernanceIsolationTests(unittest.TestCase):
    def test_missing_toolkit_does_not_replace_vector_store_by_default(self) -> None:
        vector_store = object()
        brain = SimpleNamespace(vector_db=vector_store)
        with patch.dict(os.environ, {"AGT_ENFORCEMENT_REQUIRED": "false"}), patch(
            "governance.get_runtime",
            side_effect=RuntimeError("toolkit unavailable"),
        ):
            self.assertIsNone(governance.apply_governance(brain))
        self.assertIs(brain.vector_db, vector_store)

    def test_required_mode_fails_closed_when_toolkit_is_unavailable(self) -> None:
        brain = SimpleNamespace(vector_db=object())
        with patch.dict(os.environ, {"AGT_ENFORCEMENT_REQUIRED": "true"}), patch(
            "governance.get_runtime",
            side_effect=RuntimeError("toolkit unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "toolkit unavailable"):
                governance.apply_governance(brain)

    def test_active_runtime_wraps_vector_store_without_changing_brain_api(self) -> None:
        vector_store = object()
        governed = object()
        runtime = SimpleNamespace(
            collection="quivr-demo",
            governor=SimpleNamespace(
                wrap=lambda retriever, collection: governed
            ),
        )
        brain = SimpleNamespace(vector_db=vector_store)
        with patch("governance.get_runtime", return_value=runtime):
            attached = governance.apply_governance(brain)

        self.assertIs(attached, runtime)
        self.assertIsInstance(brain.vector_db, governance.GovernedVectorStoreProxy)
        self.assertIs(brain.vector_db._vector_store, vector_store)

    def test_governed_retriever_preserves_sync_and_async_calls(self) -> None:
        calls: list[str] = []

        class VectorStore:
            def as_retriever(self, **kwargs):
                return ("raw retriever", kwargs)

        class GovernedRetriever:
            def invoke(self, query):
                calls.append(query)
                return [f"document for {query}"]

        governed = GovernedRetriever()
        runtime = SimpleNamespace(
            collection="quivr-demo",
            governor=SimpleNamespace(
                wrap=lambda retriever, collection: governed
            ),
        )
        proxy = governance.GovernedVectorStoreProxy(VectorStore(), runtime)
        retriever = proxy.as_retriever(search_kwargs={"k": 2})

        self.assertEqual(retriever.invoke("sync query"), ["document for sync query"])
        self.assertEqual(
            asyncio.run(retriever.ainvoke("async query")),
            ["document for async query"],
        )
        self.assertEqual(calls, ["sync query", "async query"])

    def test_dry_run_reports_runtime_content_scanning_without_false_deny(self) -> None:
        policy = SimpleNamespace(
            content_policies=["block_pii"],
            is_collection_allowed=lambda collection: (True, None),
        )
        runtime = SimpleNamespace(policy=policy, collection="quivr-demo")
        with patch("governance.get_runtime", return_value=runtime):
            result = governance.evaluate_governance("quivr-demo", "normal query")

        self.assertEqual(result["decision"], "allow")
        self.assertEqual(result["content_scan"], "runtime_only")
        self.assertTrue(result["warnings"])


if __name__ == "__main__":
    unittest.main()
