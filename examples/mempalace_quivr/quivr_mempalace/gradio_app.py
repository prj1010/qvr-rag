"""Gradio UI for testing MemPalace-backed RAG with Groq or NVIDIA NIM."""

from __future__ import annotations

import argparse
import os
from functools import lru_cache
from typing import Any

import gradio as gr

from .assistant import MemoryAwareAssistant
from .langchain import LangChainResponder
from .providers import (
    create_groq_model,
    create_nvidia_model,
    load_environment,
)
from .store import MempalaceMemoryStore, MemoryMatch


DEFAULT_PALACE = "~/.mempalace/palace"
PROVIDERS = ["Groq", "NVIDIA NIM"]


def _default_model(provider: str) -> str:
    if provider == "NVIDIA NIM":
        return os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b")
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


@lru_cache(maxsize=8)
def _model(provider: str, model_name: str) -> Any:
    if provider == "NVIDIA NIM":
        return create_nvidia_model(model=model_name or None)
    return create_groq_model(model=model_name or None)


def _store(wing: str, palace_path: str) -> MempalaceMemoryStore:
    return MempalaceMemoryStore(
        palace_path=palace_path.strip() or DEFAULT_PALACE,
        wing=wing.strip(),
    )


def _format_matches(matches: list[MemoryMatch]) -> str:
    if not matches:
        return "No relevant memories were found."

    lines = ["Relevant prior memories (verbatim):"]
    for index, match in enumerate(matches, start=1):
        location = "/".join(part for part in (match.wing, match.room) if part)
        lines.append(f"[{index}] {location} | similarity={match.similarity:.3f}")
        lines.append(match.content)
    return "\n".join(lines)


def recall_memories(
    question: str,
    wing: str,
    palace_path: str,
    n_results: int,
) -> str:
    """Run only the retrieval half of the RAG flow for inspection."""

    if not question.strip():
        return "Enter a question to search MemPalace."
    if not wing.strip():
        return "Enter a wing name before searching."

    try:
        matches = _store(wing, palace_path).search(
            question,
            wing=wing.strip(),
            n_results=int(n_results),
        )
        return _format_matches(matches)
    except Exception as exc:
        return f"Retrieval error: {type(exc).__name__}: {exc}"


def answer_question(
    question: str,
    history: list[list[str]] | None,
    provider: str,
    model_name: str,
    wing: str,
    palace_path: str,
    n_results: int,
) -> tuple[list[list[str]], str, str, str]:
    """Recall memories, call the selected model, and remember the turn."""

    history = history or []
    if not question.strip():
        return history, "", "Enter a question.", "Ready."
    if not wing.strip():
        return history, "", "Enter a wing name.", "Ready."

    try:
        load_environment()
        selected_model_name = model_name.strip() or _default_model(provider)
        selected_model = _model(provider, selected_model_name)
        assistant = MemoryAwareAssistant(
            store=_store(wing, palace_path),
            responder=LangChainResponder(
                selected_model,
                system_prompt=(
                    "Answer the user using relevant prior memories when useful. "
                    "If the memories do not contain the answer, say so clearly."
                ),
            ),
            wing=wing.strip(),
            n_results=int(n_results),
        )
        context = assistant.build_context(question)
        answer = assistant.ask(question)
        history = [
            *history,
            [question, answer],
        ]
        return (
            history,
            "",
            context,
            f"Connected to **{provider}** using `{selected_model_name}`.",
        )
    except Exception as exc:
        return (
            history,
            question,
            "Retrieval/model error: " + f"{type(exc).__name__}: {exc}",
            "Fix the configuration and try again.",
        )


def build_app() -> gr.Blocks:
    """Build the Gradio application without starting a web server."""

    load_environment()
    default_provider = "Groq"
    default_model = _default_model(default_provider)
    default_palace = os.getenv("MEMPALACE_PALACE_PATH", DEFAULT_PALACE)

    with gr.Blocks(title="Quivr + MemPalace RAG") as demo:
        gr.Markdown(
            "# Quivr + MemPalace RAG\n"
            "Use **Recall memories** to inspect retrieval, or ask a question "
            "to run retrieval → LLM → memory write."
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                provider = gr.Dropdown(
                    choices=PROVIDERS,
                    value=default_provider,
                    label="Model provider",
                )
                model_name = gr.Textbox(
                    value=default_model,
                    label="Model name",
                )
                wing = gr.Textbox(value="quivr-demo", label="MemPalace wing")
                palace_path = gr.Textbox(
                    value=default_palace,
                    label="Palace path",
                )
                n_results = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Memories to retrieve",
                )
                provider.change(
                    fn=lambda selected: _default_model(selected),
                    inputs=provider,
                    outputs=model_name,
                )

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation")
                question = gr.Textbox(
                    label="Question",
                    placeholder="Ask about something remembered in MemPalace...",
                    lines=2,
                )
                with gr.Row():
                    ask = gr.Button("Ask", variant="primary")
                    recall = gr.Button("Recall memories")
                    clear = gr.Button("Clear")
                retrieved = gr.Textbox(
                    label="Retrieved memory context",
                    lines=10,
                    max_lines=20,
                    interactive=False,
                )
                status = gr.Markdown("Ready.")

        ask_inputs = [
            question,
            chatbot,
            provider,
            model_name,
            wing,
            palace_path,
            n_results,
        ]
        ask_outputs = [chatbot, question, retrieved, status]
        ask.click(fn=answer_question, inputs=ask_inputs, outputs=ask_outputs)
        question.submit(fn=answer_question, inputs=ask_inputs, outputs=ask_outputs)
        recall.click(
            fn=recall_memories,
            inputs=[question, wing, palace_path, n_results],
            outputs=retrieved,
        )
        clear.click(
            fn=lambda: ([], "", "", "Ready."),
            outputs=[chatbot, question, retrieved, status],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Quivr MemPalace RAG UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    build_app().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
