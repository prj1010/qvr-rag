"""Low-overhead RAG tracing with an optional Langfuse exporter.

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
        self._langfuse_trace: Any | None = None

    def __enter__(self) -> TraceContext:
        self._token = _CURRENT_TRACE.set(self)
        self._langfuse_trace = self.recorder._start_langfuse_trace(
            self.record.name,
            self.record.attributes,
        )
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
        langfuse_span = self.recorder._start_langfuse_span(
            self._langfuse_trace,
            name,
            kind,
            record.attributes,
        )
        try:
            yield record
        except Exception as exc:
            record.status = "error"
            record.error = f"{type(exc).__name__}: {exc}"[:240]
            raise
        finally:
            record.duration_ms = (time.perf_counter() - started) * 1000
            self.record.spans.append(record)
            self.recorder._finish_langfuse_span(langfuse_span, record)

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.record.duration_ms = (time.perf_counter() - self._started) * 1000
        if exc is not None:
            self.record.status = "error"
            self.record.error = f"{exc_type.__name__}: {exc}"[:240]
        self.recorder._finish_langfuse_trace(self._langfuse_trace, self.record)
        self.recorder._store(self.record)
        if self._token is not None:
            _CURRENT_TRACE.reset(self._token)


class Observability:
    """Record recent traces locally and export them to Langfuse when configured."""

    def __init__(self, max_traces: int = 100) -> None:
        self._traces: deque[TraceRecord] = deque(maxlen=max_traces)
        self._lock = Lock()
        self._langfuse: Any | None = None
        self._langfuse_attempted = False

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

    def _get_langfuse(self) -> Any | None:
        if self._langfuse_attempted:
            return self._langfuse
        self._langfuse_attempted = True
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
        if not public_key or not secret_key:
            return None
        try:
            from langfuse import Langfuse

            host = os.getenv(
                "LANGFUSE_HOST",
                os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
            ).strip()
            self._langfuse = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                host=host,
            )
        except Exception as exc:  # telemetry must never break the product path
            print(f"Langfuse initialization skipped: {type(exc).__name__}: {exc}")
        return self._langfuse

    def _start_langfuse_trace(
        self,
        name: str,
        attributes: dict[str, Any],
    ) -> Any | None:
        client = self._get_langfuse()
        if client is None:
            return None
        try:
            return client.trace(name=name, input=attributes, metadata={"app": "mempalace-quivr"})
        except Exception as exc:
            print(f"Langfuse trace skipped: {type(exc).__name__}: {exc}")
            return None

    def _start_langfuse_span(
        self,
        trace: Any | None,
        name: str,
        kind: str,
        attributes: dict[str, Any],
    ) -> Any | None:
        if trace is None:
            return None
        try:
            if kind == "generation":
                model = attributes.get("model")
                kwargs: dict[str, Any] = {"name": name, "input": attributes}
                if model:
                    kwargs["model"] = model
                return trace.generation(**kwargs)
            return trace.span(name=name, input=attributes)
        except Exception as exc:
            print(f"Langfuse span skipped: {type(exc).__name__}: {exc}")
            return None

    @staticmethod
    def _finish_langfuse_span(span: Any | None, record: SpanRecord) -> None:
        if span is None:
            return
        try:
            span.end(
                output={"status": record.status, "duration_ms": round(record.duration_ms, 2)},
                metadata=record.attributes,
            )
        except Exception:
            try:
                span.end()
            except Exception:
                pass

    @staticmethod
    def _finish_langfuse_trace(trace: Any | None, record: TraceRecord) -> None:
        if trace is None:
            return
        try:
            trace.update(
                output={"status": record.status, "duration_ms": round(record.duration_ms, 2)},
            )
        except Exception:
            pass

    def flush(self) -> None:
        if self._langfuse is not None:
            try:
                self._langfuse.flush()
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
                "langfuse_configured": self._get_langfuse() is not None,
                "langfuse_host": os.getenv(
                    "LANGFUSE_HOST",
                    os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
                ),
                "content_capture": _bool_env("OBSERVABILITY_CAPTURE_CONTENT"),
            },
        }


OBSERVABILITY = Observability()
