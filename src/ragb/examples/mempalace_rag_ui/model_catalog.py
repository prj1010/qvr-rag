"""Curated provider/model catalog exposed to the workspace UI.

Model identifiers are intentionally kept in one backend-owned catalog so the
frontend cannot drift from the provider names accepted by the API. The list is
curated from the providers' current public catalogs; custom deployment names
remain available in the UI for self-hosted or renamed deployments.
"""

from __future__ import annotations

from typing import Any


MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "Groq": {
        "description": "Fast hosted inference with production and preview models.",
        "models": [
            {
                "id": "openai/gpt-oss-20b",
                "label": "GPT-OSS 20B",
                "details": "SLM · production · reasoning",
            },
            {
                "id": "openai/gpt-oss-120b",
                "label": "GPT-OSS 120B",
                "details": "production · reasoning",
            },
            {
                "id": "qwen/qwen3.6-27b",
                "label": "Qwen3.6 27B",
                "details": "preview · reasoning",
            },
            {
                "id": "qwen/qwen3.8-27b",
                "label": "Qwen3.8 27B",
                "details": "preview · reasoning",
            },
        ],
    },
    "NVIDIA NIM": {
        "description": "NVIDIA-hosted or self-hosted OpenAI-compatible NIM endpoints.",
        "models": [
            {
                "id": "microsoft/phi-4-mini-instruct",
                "label": "Microsoft Phi-4 Mini Instruct",
                "details": "SLM · multilingual · code",
            },
            {
                "id": "nvidia/nemotron-nano-9b-v2",
                "label": "NVIDIA Nemotron Nano 9B v2",
                "details": "SLM · reasoning",
            },
            {
                "id": "nvidia/nemotron-mini-4b-instruct",
                "label": "NVIDIA Nemotron Mini 4B",
                "details": "SLM · RAG · function calling",
            },
            {
                "id": "meta/llama-3.2-1b-instruct",
                "label": "Meta Llama 3.2 1B Instruct",
                "details": "SLM · lowest latency",
            },
            {
                "id": "meta/llama-3.2-3b-instruct",
                "label": "Meta Llama 3.2 3B Instruct",
                "details": "SLM · general purpose",
            },
            {
                "id": "nvidia/nemotron-3-nano-30b-a3b",
                "label": "NVIDIA Nemotron 3 Nano 30B",
                "details": "sparse MoE · reasoning",
            },
            {
                "id": "nvidia/nemotron-3-super-120b-a12b",
                "label": "NVIDIA Nemotron 3 Super 120B",
                "details": "large · general purpose",
            },
        ],
    },
    "Microsoft Foundry": {
        "description": "Microsoft Phi deployments through the OpenAI v1-compatible endpoint.",
        "models": [
            {
                "id": "Phi-4-mini-instruct",
                "label": "Phi-4 Mini Instruct",
                "details": "SLM · multilingual · 128K context",
            },
            {
                "id": "Phi-4-mini-reasoning",
                "label": "Phi-4 Mini Reasoning",
                "details": "SLM · reasoning · 128K context",
            },
            {
                "id": "Phi-4-multimodal-instruct",
                "label": "Phi-4 Multimodal Instruct",
                "details": "text · image · audio",
            },
            {
                "id": "Phi-4-reasoning",
                "label": "Phi-4 Reasoning",
                "details": "reasoning · 32K context",
            },
            {
                "id": "Phi-3.5-mini-instruct",
                "label": "Phi-3.5 Mini Instruct",
                "details": "SLM · efficient fallback",
            },
        ],
    },
}


def model_catalog_response() -> dict[str, Any]:
    """Return the public catalog without exposing credentials or env values."""

    return {"providers": MODEL_CATALOG}
