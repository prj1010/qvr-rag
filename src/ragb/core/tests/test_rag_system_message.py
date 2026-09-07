import unittest
from types import SimpleNamespace

from langchain_core.messages import HumanMessage, SystemMessage

from quivr_core.rag.quivr_rag_langgraph import (
    QuivrQARAGLangGraph,
    _application_instructions,
    _latest_user_message,
)


class RagSystemMessageTests(unittest.TestCase):
    def test_latest_human_message_wins_over_system_prompt(self) -> None:
        messages = [
            SystemMessage(content="Application instructions"),
            HumanMessage(content="What does the document say?"),
        ]
        self.assertEqual(
            _latest_user_message(messages), "What does the document say?"
        )

    def test_latest_human_message_supports_follow_up_messages(self) -> None:
        messages = [
            SystemMessage(content="Application instructions"),
            HumanMessage(content="First question"),
            HumanMessage(content="Follow-up question"),
        ]
        self.assertEqual(_latest_user_message(messages), "Follow-up question")

    def test_final_prompt_inputs_preserve_system_instructions_separately(self) -> None:
        messages = [
            SystemMessage(content="Use only supplied evidence."),
            HumanMessage(content="What does the document say?"),
        ]
        self.assertEqual(
            _application_instructions(messages, "Be concise."),
            "Use only supplied evidence.\n\nBe concise.",
        )
        self.assertEqual(_latest_user_message(messages), "What does the document say?")

    def test_final_rag_inputs_use_question_and_forward_system_instructions(self) -> None:
        pipeline = object.__new__(QuivrQARAGLangGraph)
        pipeline.retrieval_config = SimpleNamespace(prompt=None)
        inputs = pipeline._build_rag_prompt_inputs(
            {
                "messages": [
                    SystemMessage(content="Use only supplied evidence."),
                    HumanMessage(content="What does the document say?"),
                ],
                "files": [],
                "tasks": None,
                "chat_history": SimpleNamespace(to_list=lambda: []),
            },
            [],
        )
        self.assertEqual(inputs["task"], "What does the document say?")
        self.assertIn("Use only supplied evidence.", inputs["custom_instructions"])


if __name__ == "__main__":
    unittest.main()
