"""AICertify evidence capture and evaluation for the RAG application.

The application currently uses Quivr's LangChain 0.3 stack.  Current
AICertify releases use LangChain 1.x, so AICertify is deliberately loaded
lazily and can run in a small, separate Python environment via
``AICERTIFY_PYTHON``.  This keeps the RAG service installable while still
using the real AICertify SDK for compliance reports.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


APP_DIR = Path(__file__).resolve().parent
DEFAULT_REPORT_DIR = APP_DIR / "data" / "aicertify-reports"
SUPPORTED_FORMATS = {"markdown", "json", "pdf", "html"}
MAX_INTERACTIONS = 100
MAX_TEXT_CHARS = 20_000

_LOCK = Lock()
_INTERACTIONS: list[dict[str, Any]] = []
_LAST_EVALUATION: dict[str, Any] | None = None
_IN_PROCESS_STATUS: tuple[bool, str | None] | None = None


def _bool_setting(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _report_dir() -> Path:
    return Path(os.getenv("AICERTIFY_REPORT_DIR", str(DEFAULT_REPORT_DIR))).expanduser()


def _safe_json(value: Any) -> Any:
    """Convert AICertify/Pydantic results into JSON-safe data."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json(item) for item in value]
    if hasattr(value, "model_dump"):
        return _safe_json(value.model_dump())
    if hasattr(value, "dict"):
        return _safe_json(value.dict())
    return str(value)


def _package_version() -> str | None:
    try:
        return importlib.metadata.version("aicertify")
    except importlib.metadata.PackageNotFoundError:
        return None


def _configured_python() -> str | None:
    configured = os.getenv("AICERTIFY_PYTHON", "").strip()
    return configured or None


def _in_process_available() -> tuple[bool, str | None]:
    global _IN_PROCESS_STATUS
    if _IN_PROCESS_STATUS is not None:
        return _IN_PROCESS_STATUS
    try:
        importlib.import_module("aicertify.application")
        importlib.import_module("aicertify.regulations")
    except Exception as exc:  # AICertify imports optional evaluator/report deps eagerly.
        _IN_PROCESS_STATUS = (
            False,
            f"AICertify is not usable in this Python environment: {exc}",
        )
        return _IN_PROCESS_STATUS
    _IN_PROCESS_STATUS = (True, None)
    return _IN_PROCESS_STATUS


def _capture_enabled() -> bool:
    return _bool_setting("AICERTIFY_ENABLED", True) and _bool_setting(
        "AICERTIFY_CAPTURE_INTERACTIONS", False
    )


def record_interaction(
    input_text: str,
    output_text: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Keep a bounded, process-local evidence window when explicitly enabled."""

    if not _capture_enabled():
        return
    interaction = {
        "input_text": input_text[:MAX_TEXT_CHARS],
        "output_text": output_text[:MAX_TEXT_CHARS],
        "metadata": _safe_json(metadata or {}),
    }
    with _LOCK:
        _INTERACTIONS.append(interaction)
        del _INTERACTIONS[:-MAX_INTERACTIONS]


def captured_interactions() -> list[dict[str, Any]]:
    with _LOCK:
        return [dict(item) for item in _INTERACTIONS]


def clear_interactions() -> int:
    with _LOCK:
        count = len(_INTERACTIONS)
        _INTERACTIONS.clear()
    return count


def _evaluation_payload(
    interactions: list[dict[str, Any]], policy: str, report_format: str
) -> dict[str, Any]:
    return {
        "application_name": os.getenv("AICERTIFY_APPLICATION_NAME", "mempalace-quivr"),
        "model_name": os.getenv("AICERTIFY_MODEL_NAME", os.getenv("GROQ_MODEL", "RAG model")),
        "model_version": os.getenv("AICERTIFY_MODEL_VERSION"),
        "model_metadata": {
            "provider": os.getenv("LLM_PROVIDER", "configured in the RAG app"),
            "source": "mempalace-quivr-rag",
        },
        "interactions": interactions,
        "policy": policy,
        "report_format": report_format,
        "output_dir": str(_report_dir()),
    }


def _evaluate_in_process(payload: dict[str, Any]) -> dict[str, Any]:
    from aicertify import application, regulations

    compliance_app = application.create(
        name=payload["application_name"],
        model_name=payload["model_name"],
        model_version=payload.get("model_version"),
        model_metadata=payload.get("model_metadata"),
    )
    compliance_app.add_interactions(payload["interactions"])
    regulation_set = regulations.create("mempalace_quivr")
    regulation_set.add(payload["policy"])
    return asyncio.run(
        compliance_app.evaluate(
            regulations=regulation_set,
            generate_report=True,
            report_format=payload["report_format"],
            output_dir=payload["output_dir"],
        )
    )


def _evaluate_external(payload: dict[str, Any], python_executable: str) -> dict[str, Any]:
    runner = APP_DIR / "aicertify_runner.py"
    completed = subprocess.run(
        [python_executable, str(runner)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=int(os.getenv("AICERTIFY_TIMEOUT_SECONDS", "300")),
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(detail or f"AICertify worker exited with {completed.returncode}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("AICertify worker returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("AICertify worker returned an invalid result")
    if result.get("error"):
        raise RuntimeError(str(result["error"]))
    return result.get("results", result)


def evaluate_aicertify(
    interactions: list[dict[str, Any]] | None = None,
    policy: str | None = None,
    report_format: str | None = None,
) -> dict[str, Any]:
    """Evaluate captured RAG interactions and return AICertify results."""

    global _LAST_EVALUATION
    if not _bool_setting("AICERTIFY_ENABLED", True):
        return {"ok": False, "error": "AICertify integration is disabled by AICERTIFY_ENABLED."}

    selected_policy = (policy or os.getenv("AICERTIFY_POLICY", "eu_ai_act")).strip()
    selected_format = (report_format or os.getenv("AICERTIFY_REPORT_FORMAT", "markdown")).lower()
    if selected_format not in SUPPORTED_FORMATS:
        return {"ok": False, "error": f"Unsupported report format: {selected_format}."}

    selected_interactions = interactions if interactions is not None else captured_interactions()
    if not selected_interactions:
        return {
            "ok": False,
            "error": "No captured interactions. Set AICERTIFY_CAPTURE_INTERACTIONS=true and ask a question first.",
        }

    payload = _evaluation_payload(selected_interactions, selected_policy, selected_format)
    try:
        configured_python = _configured_python()
        if configured_python:
            results = _evaluate_external(payload, configured_python)
            execution = f"external:{configured_python}"
        else:
            available, reason = _in_process_available()
            if not available:
                raise RuntimeError(
                    f"{reason}. Set AICERTIFY_PYTHON to a Python environment created with 'pip install aicertify'."
                )
            results = _evaluate_in_process(payload)
            execution = "in_process"
        response = {
            "ok": True,
            "policy": selected_policy,
            "report_format": selected_format,
            "interaction_count": len(selected_interactions),
            "execution": execution,
            "results": _safe_json(results),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        response = {
            "ok": False,
            "policy": selected_policy,
            "report_format": selected_format,
            "interaction_count": len(selected_interactions),
            "error": f"{type(exc).__name__}: {exc}",
        }
    with _LOCK:
        _LAST_EVALUATION = response
    return response


def aicertify_snapshot() -> dict[str, Any]:
    """Return admin-safe AICertify configuration and last-run metadata."""

    enabled = _bool_setting("AICERTIFY_ENABLED", True)
    configured_python = _configured_python()
    in_process_reason = None
    if not configured_python:
        _, in_process_reason = _in_process_available()
    with _LOCK:
        last_evaluation = _safe_json(_LAST_EVALUATION)
        interaction_count = len(_INTERACTIONS)
    return {
        "enabled": enabled,
        "package_version": _package_version(),
        "capture_interactions": _bool_setting("AICERTIFY_CAPTURE_INTERACTIONS", False),
        "captured_interactions": interaction_count,
        "policy": os.getenv("AICERTIFY_POLICY", "eu_ai_act"),
        "report_format": os.getenv("AICERTIFY_REPORT_FORMAT", "markdown"),
        "report_dir": str(_report_dir()),
        "python": configured_python or sys.executable,
        "execution_mode": "external" if configured_python else "in_process",
        "availability_error": in_process_reason,
        "last_evaluation": last_evaluation,
    }


__all__ = [
    "aicertify_snapshot",
    "captured_interactions",
    "clear_interactions",
    "evaluate_aicertify",
    "record_interaction",
]
