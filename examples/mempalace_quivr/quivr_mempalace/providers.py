"""Provider constructors that read credentials from a local ``.env`` file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_environment(env_file: str | Path | None = None) -> None:
    """Load provider settings from ``.env`` without replacing shell variables."""

    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            "Install the model dependencies first: "
            'python -m pip install -e ".[models]"'
        ) from exc

    load_dotenv(dotenv_path=env_file, override=False)


def _required_environment(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing {name}. Add it to .env or set it in the CMD session."
        )
    return value


def create_groq_model(
    *,
    model: str | None = None,
    temperature: float = 0.0,
    env_file: str | Path | None = None,
) -> Any:
    """Create a LangChain ``ChatGroq`` model from the local environment."""

    load_environment(env_file)
    _required_environment("GROQ_API_KEY")

    from langchain_groq import ChatGroq

    return ChatGroq(
        model=model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=temperature,
    )


def create_nvidia_model(
    *,
    model: str | None = None,
    temperature: float = 0.0,
    env_file: str | Path | None = None,
) -> Any:
    """Create a LangChain ``ChatNVIDIA`` model for hosted or local NIM."""

    load_environment(env_file)

    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    base_url = os.getenv("NVIDIA_BASE_URL", "").strip()
    if not api_key and not base_url:
        raise RuntimeError(
            "Set NVIDIA_API_KEY for hosted NIM or NVIDIA_BASE_URL for a local NIM."
        )

    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    kwargs: dict[str, Any] = {
        "model": model or os.getenv(
            "NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b"
        ),
        "temperature": temperature,
    }
    if api_key:
        kwargs["nvidia_api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return ChatNVIDIA(**kwargs)
