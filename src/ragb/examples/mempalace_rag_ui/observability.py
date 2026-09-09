"""Low-overhead RAG tracing with optional Pydantic Logfire export.

The local recorder intentionally stores metadata rather than prompts, answers, or
uploaded document contents by default.  This makes the admin console useful in a
hosted deployment without turning it into a second data store for user content.
Set ``OBSERVABILITY_CAPTURE_CONTENT=true`` only when that trade-off is acceptable.
"""

from __future__ import annotations

import contextlib
import math
import os
import time
from collections import deque
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any, Iterator
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _configure_logfire() -> Any | None:
    """Configure Logfire without making telemetry a product-path dependency."""

    # LangChain/LangGraph expose their OpenTelemetry spans through these
    # settings. They must be present before LangChain is imported.
    for name in ("LANGSMITH_OTEL_ENABLED", "LANGSMITH_OTEL_ONLY", "LANGSMITH_TRACING"):
        os.environ.setdefault(name, "true")

    try:
        import logfire
    except ImportError:
        return None

    send_setting: bool | str = "if-token-present"
    if os.getenv("LOGFIRE_SEND_TO_LOGFIRE") is not None:
        send_setting = _bool_env("LOGFIRE_SEND_TO_LOGFIRE")

    try:
        logfire.configure(
            send_to_logfire=send_setting,
            service_name=os.getenv("LOGFIRE_SERVICE_NAME", "quivr-mempalace-rag"),
        )
        instrument_pydantic = getattr(logfire, "instrument_pydantic", None)
        if callable(instrument_pydantic) and _bool_env(
            "LOGFIRE_INSTRUMENT_PYDANTIC", True
        ):
            instrument_pydantic()
        return logfire
    except Exception as exc:  # telemetry must never break the product path
        print(f"Logfire initialization skipped: {type(exc).__name__}: {exc}")
        return None


def _safe_value(value: Any) -> Any:
    if isinstance(value, str):
        return value[:240]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value[:20]]
    return str(value)[:240]


def _safe_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    return {str(key): _safe_value(value) for key, value in (attributes or {}).items()}


def content_metadata(value: str, *, label: str = "text") -> dict[str, Any]:
    """Return safe content metadata, optionally including a short preview."""

    metadata: dict[str, Any] = {f"{label}_chars": len(value)}
    if _bool_env("OBSERVABILITY_CAPTURE_CONTENT"):
        metadata[f"{label}_preview"] = value[:240]
    return metadata


@dataclass
class SpanRecord:
    span_id: str
    name: str
    kind: str
    started_at: str
    duration_ms: float = 0.0
    status: str = "ok"
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "name": self.name,
            "kind": self.kind,
            "started_at": self.started_at,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "attributes": self.attributes,
            "error": self.error,
        }


@dataclass
class TraceRecord:
    trace_id: str
    name: str
    started_at: str
    attributes: dict[str, Any] = field(default_factory=dict)
    spans: list[SpanRecord] = field(default_factory=list)
    duration_ms: float = 0.0
    status: str = "ok"
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "started_at": self.started_at,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "attributes": self.attributes,
            "spans": [span.as_dict() for span in self.spans],
            "error": self.error,
        }


_CURRENT_TRACE: ContextVar[TraceContext | None] = ContextVar("current_trace", default=None)


class TraceContext:
    def __init__(
        self,
        recorder: Observability,
        name: str,
        attributes: dict[str, Any] | None,
    ) -> None:
        self.recorder = recorder
        self.record = TraceRecord(
            trace_id=str(uuid4()),
            name=name,
            started_at=datetime.now(UTC).isoformat(),
            attributes=_safe_attributes(attributes),
        )
        self._started = time.perf_counter()
        self._token: Token[TraceContext | None] | None = None
        self._logfire_trace: Any | None = None

    def __enter__(self) -> TraceContext:
        self._token = _CURRENT_TRACE.set(self)
        self._logfire_trace = self.recorder._start_logfire_span(
            self.record.name, self.record.attributes
        )
        self.recorder._enter_logfire_context(self._logfire_trace)
        return self

    @contextlib.contextmanager
    def span(
        self,
        name: str,
        kind: str = "span",
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[SpanRecord]:
        record = SpanRecord(
            span_id=str(uuid4()),
            name=name,
            kind=kind,
            started_at=datetime.now(UTC).isoformat(),
            attributes=_safe_attributes(attributes),
        )
        started = time.perf_counter()
        logfire_span = self.recorder._start_logfire_span(name, record.attributes)
        self.recorder._enter_logfire_context(logfire_span)
        error_info: tuple[type[BaseException] | None, BaseException | None, Any] = (
            None,
            None,
            None,
        )
        try:
            yield record
        except BaseException as exc:
            record.status = "error"
            record.error = f"{type(exc).__name__}: {exc}"[:240]
            error_info = (type(exc), exc, exc.__traceback__)
            raise
        finally:
            record.duration_ms = (time.perf_counter() - started) * 1000
            self.record.spans.append(record)
            self.recorder._exit_logfire_context(logfire_span, error_info)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.record.duration_ms = (time.perf_counter() - self._started) * 1000
        if exc is not None:
            self.record.status = "error"
            self.record.error = f"{exc_type.__name__}: {exc}"[:240]
        self.recorder._exit_logfire_context(
            self._logfire_trace,
            (exc_type, exc, traceback),
        )
        self.recorder._store(self.record)
        if self._token is not None:
            _CURRENT_TRACE.reset(self._token)


class Observability:
    """Record recent traces locally and export them to Pydantic Logfire."""

    def __init__(self, max_traces: int = 100) -> None:
        self._traces: deque[TraceRecord] = deque(maxlen=max_traces)
        self._lock = Lock()
        self._logfire = _configure_logfire()

    def start_trace(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> TraceContext:
        return TraceContext(self, name, attributes)

    def span(
        self,
        name: str,
        kind: str = "span",
        attributes: dict[str, Any] | None = None,
    ) -> contextlib.AbstractContextManager[Any]:
        trace = _CURRENT_TRACE.get()
        if trace is None:
            return contextlib.nullcontext()
        return trace.span(name, kind, attributes)

    def _store(self, trace: TraceRecord) -> None:
        with self._lock:
            self._traces.appendleft(trace)

    def _start_logfire_span(
        self, name: str, attributes: dict[str, Any]
    ) -> Any | None:
        if self._logfire is None:
            return None
        try:
            return self._logfire.span(name, **attributes)
        except Exception as exc:  # telemetry must never break the product path
            print(f"Logfire span skipped: {type(exc).__name__}: {exc}")
            return None

    @staticmethod
    def _enter_logfire_context(context: Any | None) -> None:
        if context is None:
            return
        try:
            context.__enter__()
        except Exception:
            pass

    @staticmethod
    def _exit_logfire_context(
        context: Any | None,
        error_info: tuple[type[BaseException] | None, BaseException | None, Any],
    ) -> None:
        if context is None:
            return
        try:
            context.__exit__(*error_info)
        except Exception:
            pass

    def instrument_fastapi(self, app: Any) -> None:
        """Instrument FastAPI when the optional Logfire SDK is available."""

        if self._logfire is None:
            return
        try:
            self._logfire.instrument_fastapi(app)
        except Exception as exc:
            print(f"Logfire FastAPI instrumentation skipped: {type(exc).__name__}: {exc}")

    def flush(self) -> None:
        if self._logfire is None:
            return
        force_flush = getattr(self._logfire, "force_flush", None)
        if callable(force_flush):
            try:
                force_flush()
            except Exception:
                pass

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            traces = [trace.as_dict() for trace in self._traces]
        durations = [trace["duration_ms"] for trace in traces]
        errors = sum(trace["status"] == "error" for trace in traces)
        ordered_durations = sorted(durations)
        p95 = 0.0
        if ordered_durations:
            p95 = ordered_durations[max(0, math.ceil(len(ordered_durations) * 0.95) - 1)]
        operations: dict[str, int] = {}
        for trace in traces:
            operations[trace["name"]] = operations.get(trace["name"], 0) + 1
        return {
            "summary": {
                "total_traces": len(traces),
                "error_traces": errors,
                "error_rate": round(errors / len(traces), 3) if traces else 0.0,
                "avg_latency_ms": round(sum(durations) / len(durations), 2) if durations else 0.0,
                "p95_latency_ms": round(p95, 2),
                "operations": operations,
            },
            "traces": traces,
            "integrations": {
                "logfire_configured": self._logfire is not None,
                "logfire_send_to_logfire": _bool_env(
                    "LOGFIRE_SEND_TO_LOGFIRE", bool(os.getenv("LOGFIRE_TOKEN"))
                ),
                "logfire_project_url": os.getenv("LOGFIRE_PROJECT_URL", ""),
                "content_capture": _bool_env("OBSERVABILITY_CAPTURE_CONTENT"),
            },
        }


OBSERVABILITY = Observability()
