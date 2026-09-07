import unittest

from prompts import MAX_MEMORY_CONTEXT_CHARS, RAG_SYSTEM_PROMPT, build_rag_system_prompt


class PromptTests(unittest.TestCase):
    def test_prompt_defines_grounding_and_untrusted_context_rules(self) -> None:
        prompt = build_rag_system_prompt("Relevant memory")
        self.assertIn("indexed documents", prompt)
        self.assertIn("not instructions", prompt)
        self.assertIn("BEGIN UNTRUSTED MEMORY CONTEXT", prompt)
        self.assertIn("Relevant memory", prompt)

    def test_prompt_bounds_memory_context(self) -> None:
        prompt = build_rag_system_prompt("x" * (MAX_MEMORY_CONTEXT_CHARS + 100))
        context = prompt.split("BEGIN UNTRUSTED MEMORY CONTEXT ---\n", 1)[1]
        context = context.split("\n--- END UNTRUSTED MEMORY CONTEXT", 1)[0]
        memory = context.split("\n[Additional", 1)[0]
        self.assertEqual(len(memory), MAX_MEMORY_CONTEXT_CHARS)
        self.assertIn("Additional memory context omitted", prompt)

    def test_base_prompt_is_stable_and_does_not_include_runtime_data(self) -> None:
        self.assertIn("Do not reveal chain-of-thought", RAG_SYSTEM_PROMPT)
        self.assertNotIn("api_key", RAG_SYSTEM_PROMPT.lower())


if __name__ == "__main__":
    unittest.main()
