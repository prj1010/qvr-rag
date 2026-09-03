from __future__ import annotations

import unittest
from unittest import mock
import sys
import types
from pathlib import Path

from quivr_mempalace import (
    LangChainResponder,
    MemoryAwareAssistant,
    MemoryMatch,
    MemoryRecord,
    QuivrBrainResponder,
    stable_memory_id,
)


class FakeStore:
    def __init__(self) -> None:
        self.added: list[MemoryRecord] = []

    def add(self, memory: MemoryRecord) -> str:
        self.added.append(memory)
        return stable_memory_id(memory)

    def search(
        self,
        query: str,
        *,
        wing: str | None = None,
        room: str | None = None,
        n_results: int = 5,
    ) -> list[MemoryMatch]:
        return [
            MemoryMatch(
                content="I prefer concise answers.",
                wing=wing or "demo",
                room=room or "preferences",
                similarity=0.91,
            )
        ]

    def wake_up(self, *, wing: str | None = None) -> str:
        return ""


class MemoryModelTests(unittest.TestCase):
    def test_langchain_responder_invokes_model_and_returns_content(self) -> None:
        class Model:
            def __init__(self) -> None:
                self.prompt = ""

            def invoke(self, prompt: str):
                self.prompt = prompt
                return types.SimpleNamespace(content="langchain answer")

        model = Model()
        responder = LangChainResponder(model, system_prompt="Answer clearly.")

        answer = responder("What do I prefer?", "I prefer concise answers.")

        self.assertEqual(answer, "langchain answer")
        self.assertIn("Answer clearly.", model.prompt)
        self.assertIn("I prefer concise answers.", model.prompt)

    def test_memory_id_is_deterministic_and_location_sensitive(self) -> None:
        memory = MemoryRecord("same text", "one", "room")
        self.assertEqual(stable_memory_id(memory), stable_memory_id(memory))
        self.assertNotEqual(
            stable_memory_id(memory),
            stable_memory_id(MemoryRecord("same text", "two", "room")),
        )

    def test_assistant_recalls_and_remembers_turn(self) -> None:
        store = FakeStore()
        seen: dict[str, str] = {}

        def responder(question: str, context: str) -> str:
            seen["question"] = question
            seen["context"] = context
            return "Use a concise Python example."

        assistant = MemoryAwareAssistant(
            store=store,
            responder=responder,
            wing="demo",
        )
        answer = assistant.ask("How should you answer me?")

        self.assertEqual(answer, "Use a concise Python example.")
        self.assertEqual(seen["question"], "How should you answer me?")
        self.assertIn("I prefer concise answers.", seen["context"])
        self.assertEqual(len(store.added), 1)
        self.assertIn("User: How should you answer me?", store.added[0].content)

    def test_record_rejects_empty_required_fields(self) -> None:
        with self.assertRaises(ValueError):
            MemoryRecord("", "demo", "room")

    def test_quivr_responder_injects_memory_context(self) -> None:
        class Response:
            answer = "Answer from Quivr"

        class Brain:
            def ask(self, **kwargs):
                self.kwargs = kwargs
                return Response()

        brain = Brain()
        responder = QuivrBrainResponder(brain, system_prompt="Be helpful.")
        answer = responder("What do I prefer?", "I prefer concise answers.")

        self.assertEqual(answer, "Answer from Quivr")
        self.assertEqual(brain.kwargs["question"], "What do I prefer?")
        self.assertIn("I prefer concise answers.", brain.kwargs["system_prompt"])

    def test_mempalace_adapter_forwards_drawer_and_search_shapes(self) -> None:
        from quivr_mempalace import MempalaceMemoryStore

        class Collection:
            def upsert(self, **kwargs):
                self.upsert_kwargs = kwargs

        collection = Collection()
        config_module = types.ModuleType("mempalace.config")

        class Config:
            collection_name = "mempalace_drawers"

        config_module.MempalaceConfig = Config
        palace_module = types.ModuleType("mempalace.palace")
        palace_module.get_collection = lambda *args, **kwargs: collection
        search_module = types.ModuleType("mempalace.searcher")
        search_module.search_memories = lambda **kwargs: {
            "results": [
                {
                    "text": "A remembered choice.",
                    "wing": "demo",
                    "room": "decisions",
                    "source_file": "",
                    "similarity": 0.88,
                }
            ]
        }
        mempalace_module = types.ModuleType("mempalace")
        mempalace_module.__path__ = []

        fake_modules = {
            "mempalace": mempalace_module,
            "mempalace.config": config_module,
            "mempalace.palace": palace_module,
            "mempalace.searcher": search_module,
        }
        with mock.patch.dict(sys.modules, fake_modules):
            palace_path = Path.cwd()
            store = MempalaceMemoryStore(palace_path, wing="demo")
            drawer_id = store.add(
                MemoryRecord("A remembered choice.", "demo", "decisions")
            )
            matches = store.search("What choice did I make?")

        self.assertTrue(drawer_id.startswith("quivr-memory-"))
        self.assertEqual(collection.upsert_kwargs["documents"], ["A remembered choice."])
        self.assertEqual(matches[0].content, "A remembered choice.")


if __name__ == "__main__":
    unittest.main()
