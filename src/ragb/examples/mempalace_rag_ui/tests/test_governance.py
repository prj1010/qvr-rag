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


if __name__ == "__main__":
    unittest.main()
