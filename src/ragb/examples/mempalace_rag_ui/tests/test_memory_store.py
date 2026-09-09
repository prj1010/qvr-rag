import tempfile
import unittest
from pathlib import Path

from memory_search import hashed_embedding
from memory_store import WorkspaceStore


class MemoryStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = WorkspaceStore(Path(self.tempdir.name) / "workspace.duckdb")

    def tearDown(self) -> None:
        self.store.close()
        self.tempdir.cleanup()

    def test_add_turn_round_trip_and_search(self) -> None:
        self.store.add_turn(
            "What is the refund policy?",
            "Refunds are issued within five days.",
            question_emb=hashed_embedding("What is the refund policy?"),
            answer_emb=hashed_embedding("Refunds are issued within five days."),
        )
        messages = self.store.list_messages()
        self.assertEqual(
            [item["role"] for item in messages], ["user", "assistant"]
        )
        matches = self.store.search(
            "refund policy",
            hashed_embedding("refund policy"),
            threshold=0.2,
            limit=5,
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].matched_field, "question")

    def test_refresh_restores_messages_until_explicit_clear(self) -> None:
        self.store.add_turn("Hello", "Hi there.")
        self.store.set_meta("model_name", "openai/gpt-oss-20b")
        self.store.close()
        restored = WorkspaceStore(Path(self.tempdir.name) / "workspace.duckdb")
        self.store = restored
        self.assertEqual(len(restored.list_messages()), 2)
        self.assertEqual(restored.get_meta("model_name"), "openai/gpt-oss-20b")
        restored.clear_memories()
        self.assertEqual(restored.list_messages(), [])
        self.assertEqual(restored.get_meta("model_name"), "openai/gpt-oss-20b")
        restored.clear_workspace()
        self.assertIsNone(restored.get_meta("model_name"))


if __name__ == "__main__":
    unittest.main()
