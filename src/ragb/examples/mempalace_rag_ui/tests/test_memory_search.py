import unittest

from memory_search import hashed_embedding, rank_memories, score_memory


class MemorySearchTests(unittest.TestCase):
    def test_hashed_embedding_is_stable_and_normalized(self) -> None:
        left = hashed_embedding("What is the refund policy?")
        right = hashed_embedding("What is the refund policy?")
        self.assertEqual(left, right)
        self.assertAlmostEqual(sum(value * value for value in left), 1.0, places=5)

    def test_identifies_question_match_with_regex_and_cosine(self) -> None:
        query = "refund policy for annual plans"
        query_vector = hashed_embedding(query)
        question = "What is the refund policy for annual plans?"
        answer = "Shipping takes three days after checkout."
        match = score_memory(
            query,
            query_vector,
            "m1",
            question,
            answer,
            hashed_embedding(question),
            hashed_embedding(answer),
            threshold=0.2,
        )
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.matched_field, "question")
        self.assertIn("refund", match.regex_hits)
        self.assertGreater(match.cosine_question, match.cosine_answer)

    def test_identifies_answer_match(self) -> None:
        query = "three day shipping"
        query_vector = hashed_embedding(query)
        question = "How do refunds work?"
        answer = "Standard shipping takes three days after checkout."
        match = score_memory(
            query,
            query_vector,
            "m2",
            question,
            answer,
            hashed_embedding(question),
            hashed_embedding(answer),
            threshold=0.2,
        )
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.matched_field, "answer")
        self.assertGreater(match.cosine_answer, match.cosine_question)

    def test_threshold_drops_unrelated_memory(self) -> None:
        query = "kubernetes rollout strategy"
        match = score_memory(
            query,
            hashed_embedding(query),
            "m3",
            "What is the cafeteria menu?",
            "Soup and salad are served at noon.",
            hashed_embedding("What is the cafeteria menu?"),
            hashed_embedding("Soup and salad are served at noon."),
            threshold=0.55,
        )
        self.assertIsNone(match)

    def test_rank_memories_respects_limit(self) -> None:
        query = "refund policy"
        rows = [
            {
                "id": "a",
                "question": "What is the refund policy?",
                "answer": "Refunds are issued within five days.",
                "question_emb": hashed_embedding("What is the refund policy?"),
                "answer_emb": hashed_embedding("Refunds are issued within five days."),
            },
            {
                "id": "b",
                "question": "Office hours?",
                "answer": "The office opens at nine.",
                "question_emb": hashed_embedding("Office hours?"),
                "answer_emb": hashed_embedding("The office opens at nine."),
            },
        ]
        matches = rank_memories(
            query, hashed_embedding(query), rows, threshold=0.2, limit=1
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].memory_id, "a")


if __name__ == "__main__":
    unittest.main()
