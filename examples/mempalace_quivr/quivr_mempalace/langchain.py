"""Adapters for LangChain chat and text models."""

from typing import Any


class LangChainResponder:
    """Turn any LangChain model with ``invoke`` into a memory-aware responder.

    The adapter accepts the model as an object instead of importing a provider
    package. This keeps provider imports optional while supporting ChatGroq,
    ChatNVIDIA, HuggingFacePipeline, ChatHuggingFace, and other compatible
    LangChain models.
    """

    def __init__(self, model: Any, *, system_prompt: str = "") -> None:
        invoke = getattr(model, "invoke", None)
        if not callable(invoke):
            raise TypeError("model must expose a callable invoke(prompt) method")
        self.model = model
        self.system_prompt = system_prompt.strip()

    def __call__(self, question: str, context: str) -> str:
        sections = []
        if self.system_prompt:
            sections.append(self.system_prompt)
        if context:
            sections.append(
                "Relevant memories:\n"
                "Use these memories as context, but do not invent details:\n"
                f"{context}"
            )
        sections.append(f"User question:\n{question}")

        result = self.model.invoke("\n\n".join(sections))
        content = getattr(result, "content", result)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
            if text_parts:
                return "\n".join(text_parts)
        return str(content)
