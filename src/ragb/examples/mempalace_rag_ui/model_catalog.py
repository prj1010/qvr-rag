"""Curated provider/model catalog exposed to the workspace UI.

Model identifiers are intentionally kept in one backend-owned catalog so the
frontend cannot drift from the provider names accepted by the API. Each model
declares its context window so memory packing can stay inside the selected
model's budget instead of dumping the full palace into the prompt.
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
                "context_window": 131072,
                "max_output_tokens": 8192,
            },
            {
                "id": "openai/gpt-oss-120b",
                "label": "GPT-OSS 120B",
                "details": "production · reasoning",
                "context_window": 131072,
                "max_output_tokens": 8192,
            },
            {
                "id": "qwen/qwen3.6-27b",
                "label": "Qwen3.6 27B",
                "details": "preview · reasoning",
                "context_window": 262144,
                "max_output_tokens": 8192,
            },
            {
                "id": "qwen/qwen3.8-27b",
                "label": "Qwen3.8 27B",
                "details": "preview · reasoning",
                "context_window": 262144,
                "max_output_tokens": 8192,
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
                "context_window": 131072,
                "max_output_tokens": 4096,
            },
            {
                "id": "nvidia/nemotron-nano-9b-v2",
                "label": "NVIDIA Nemotron Nano 9B v2",
                "details": "SLM · reasoning",
                "context_window": 131072,
                "max_output_tokens": 4096,
            },
            {
                "id": "nvidia/nemotron-mini-4b-instruct",
                "label": "NVIDIA Nemotron Mini 4B",
                "details": "SLM · RAG · function calling",
                "context_window": 4096,
                "max_output_tokens": 2048,
            },
            {
                "id": "meta/llama-3.2-1b-instruct",
                "label": "Meta Llama 3.2 1B Instruct",
                "details": "SLM · lowest latency",
                "context_window": 131072,
                "max_output_tokens": 4096,
            },
            {
                "id": "meta/llama-3.2-3b-instruct",
                "label": "Meta Llama 3.2 3B Instruct",
                "details": "SLM · general purpose",
                "context_window": 131072,
                "max_output_tokens": 4096,
            },
            {
                "id": "nvidia/nemotron-3-nano-30b-a3b",
                "label": "NVIDIA Nemotron 3 Nano 30B",
                "details": "sparse MoE · reasoning",
                "context_window": 131072,
                "max_output_tokens": 8192,
            },
            {
                "id": "nvidia/nemotron-3-super-120b-a12b",
                "label": "NVIDIA Nemotron 3 Super 120B",
                "details": "large · general purpose",
                "context_window": 262144,
                "max_output_tokens": 8192,
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
                "context_window": 131072,
                "max_output_tokens": 4096,
            },
            {
                "id": "Phi-4-mini-reasoning",
                "label": "Phi-4 Mini Reasoning",
                "details": "SLM · reasoning · 128K context",
                "context_window": 131072,
                "max_output_tokens": 4096,
            },
            {
                "id": "Phi-4-multimodal-instruct",
                "label": "Phi-4 Multimodal Instruct",
                "details": "text · image · audio",
                "context_window": 131072,
                "max_output_tokens": 4096,
            },
            {
                "id": "Phi-4-reasoning",
                "label": "Phi-4 Reasoning",
                "details": "reasoning · 32K context",
                "context_window": 32768,
                "max_output_tokens": 4096,
            },
            {
                "id": "Phi-3.5-mini-instruct",
                "label": "Phi-3.5 Mini Instruct",
                "details": "SLM · efficient fallback",
                "context_window": 131072,
                "max_output_tokens": 4096,
            },
        ],
    },
}


def _all_models() -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for provider in MODEL_CATALOG.values():
        models.extend(provider.get("models") or [])
    return models


def find_model(model_id: str) -> dict[str, Any] | None:
    needle = (model_id or "").strip()
    for model in _all_models():
        if model.get("id") == needle:
            return model
    return None


def context_window_for(model_id: str, default: int = 32768) -> int:
    model = find_model(model_id)
    if model and int(model.get("context_window") or 0) > 0:
        return int(model["context_window"])
    return default


def max_output_tokens_for(model_id: str, default: int = 4096) -> int:
    model = find_model(model_id)
    if model and int(model.get("max_output_tokens") or 0) > 0:
        return int(model["max_output_tokens"])
    return default


def model_catalog_response() -> dict[str, Any]:
    """Return the public catalog without exposing credentials or env values."""

    return {"providers": MODEL_CATALOG}
