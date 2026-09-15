import unittest

from context_policy import estimate_tokens, model_budgets, pack_memory_context
from memory_search import MemoryMatch
from model_catalog import context_window_for, max_output_tokens_for


def _match(memory_id: str, question: str, answer: str, score: float) -> MemoryMatch:
    return MemoryMatch(
        memory_id=memory_id,
        question=question,
        answer=answer,
        matched_field="answer",
        cosine_question=score,
        cosine_answer=score,
        cosine=score,
        regex_hits=("policy",),
        score=score,
        explanation="test match",
    )


class ContextPolicyTests(unittest.TestCase):
    def test_smaller_models_get_a_tighter_memory_budget(self) -> None:
        mini = model_budgets("nvidia/nemotron-mini-4b-instruct")
        large = model_budgets("openai/gpt-oss-20b")
        self.assertLess(mini["context_window"], large["context_window"])
        self.assertLess(mini["memory_budget"], large["memory_budget"])
        self.assertEqual(
            mini["context_window"],
            context_window_for("nvidia/nemotron-mini-4b-instruct"),
        )
        self.assertEqual(
            large["output_reserve"], max_output_tokens_for("openai/gpt-oss-20b")
        )

    def test_low_similarity_memories_are_denied(self) -> None:
        packed = pack_memory_context(
            [_match("low", "unrelated", "noise", 0.05)],
            model_id="openai/gpt-oss-20b",
            threshold=0.42,
        )
        self.assertEqual(packed.kept, [])
        self.assertEqual(len(packed.dropped), 1)
        self.assertIn("deny_low_similarity", packed.applied_rules)

    def test_overflow_is_summarized_instead_of_dumped(self) -> None:
        matches = [
            _match(
                f"m{index}",
                f"Question {index} about the refund policy and annual billing cycle",
                "A" * 800,
                0.9,
            )
            for index in range(40)
        ]
        packed = pack_memory_context(
            matches,
            model_id="nvidia/nemotron-mini-4b-instruct",
            threshold=0.2,
        )
        self.assertTrue(packed.kept)
        self.assertTrue(packed.dropped)
        self.assertTrue(packed.summarized)
        self.assertIn("summarize_over_budget", packed.applied_rules)
        self.assertLessEqual(
            packed.token_usage["memory"], packed.budget["memory_budget"] * 2
        )
        self.assertIn("Running summary of earlier conversation", packed.memory_block)
        self.assertLess(
            estimate_tokens(packed.memory_block),
            estimate_tokens("\n".join(item.answer for item in matches)),
        )


if __name__ == "__main__":
    unittest.main()
