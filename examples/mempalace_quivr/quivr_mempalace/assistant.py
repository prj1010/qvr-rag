"""A model-agnostic assistant wrapper that adds MemPalace recall."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .store import MemoryRecord, MemoryStore, MemoryMatch


Responder = Callable[[str, str], str]


@dataclass(frozen=True)
class PromptContext:
    """The question plus the memory context sent to a model provider."""

    question: str
    context: str


class MemoryAwareAssistant:
    """Recall relevant memories, call a responder, then remember the turn."""

    def __init__(
        self,
        *,
        store: MemoryStore,
        responder: Responder,
        wing: str,
        room: str = "conversation",
        n_results: int = 5,
        remember_turns: bool = True,
    ) -> None:
        if n_results < 1:
            raise ValueError("n_results must be at least 1")
        self.store = store
        self.responder = responder
        self.wing = wing
        self.room = room
        self.n_results = n_results
        self.remember_turns = remember_turns

    def recall(self, question: str) -> list[MemoryMatch]:
        return self.store.search(
            question,
            wing=self.wing,
            n_results=self.n_results,
        )

    def build_context(self, question: str) -> str:
        """Render relevant memories as a compact, inspectable context block."""

        matches = self.recall(question)
        if not matches:
            return "No relevant prior memories were found."

        lines = ["Relevant prior memories (verbatim):"]
        for index, match in enumerate(matches, start=1):
            location = "/".join(part for part in (match.wing, match.room) if part)
            score = f"{match.similarity:.3f}"
            lines.append(f"[{index}] {location} | similarity={score}")
            lines.append(match.content)
        return "\n".join(lines)

    def build_prompt(self, question: str) -> PromptContext:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        return PromptContext(question=question, context=self.build_context(question))

    def ask(self, question: str) -> str:
        """Ask the configured responder with recalled context and remember it."""

        prompt = self.build_prompt(question)
        answer = self.responder(prompt.question, prompt.context)
        if not isinstance(answer, str):
            raise TypeError("responder must return a string")

        if self.remember_turns:
            self.store.add(
                MemoryRecord(
                    content=f"User: {question}\nAssistant: {answer}",
                    wing=self.wing,
                    room=self.room,
                )
            )
        return answer
