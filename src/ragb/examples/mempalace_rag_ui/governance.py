"""Microsoft Agent Governance Toolkit integration for the Quivr RAG path.

The forked toolkit deliberately disables its asynchronous ``ainvoke`` method
on governed retrievers so callers cannot accidentally bypass governance. Quivr
uses LangChain's async retriever API internally, so this module adapts that
call to the toolkit's synchronous ``invoke`` method in a worker thread.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from starlette.concurrency import run_in_threadpool


APP_DIR = Path(__file__).resolve().parent
DEFAULT_COLLECTION = "quivr-demo"
DEFAULT_AUDIT_PATH = APP_DIR / "data" / "agent-rag-audit.jsonl"
_RUNTIME_LOCK = Lock()
_RUNTIME: GovernanceRuntime | None = None
_RUNTIME_ERROR: str | None = None


def _csv_setting(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _bool_setting(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _int_setting(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer.") from exc


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True)
class GovernanceRuntime:
    governor: Any
    policy: Any
    agent_id: str
    collection: str
    audit_path: Path
    core_version: str | None
    rag_version: str | None

    @property
    def capabilities(self) -> list[str]:
        return [
            "retrieval_collection_acl",
            "retrieval_rate_limiting",
            "pii_scanning",
            "prompt_injection_scanning",
            "privacy_safe_audit_hashes",
            "langchain_async_adapter",
        ]


def _create_runtime() -> GovernanceRuntime:
    try:
        from agent_rag_governance import RAGGovernor, RAGPolicy
    except ImportError as exc:
        raise RuntimeError(
            "Agent Governance Toolkit is not installed. Run the pinned fork "
            "installation from requirements.txt."
        ) from exc

    agent_id = os.getenv("AGT_AGENT_ID", "mempalace-quivr").strip() or "mempalace-quivr"
    collection = os.getenv("AGT_COLLECTION", DEFAULT_COLLECTION).strip() or DEFAULT_COLLECTION
    allowed_collections = _csv_setting("AGT_ALLOWED_COLLECTIONS", collection)
    denied_collections = _csv_setting("AGT_DENIED_COLLECTIONS")
    audit_path = Path(
        os.getenv("AGT_AUDIT_LOG_PATH", str(DEFAULT_AUDIT_PATH))
    ).expanduser()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    content_policies = _csv_setting(
        "AGT_CONTENT_POLICIES", "block_pii,block_injections"
    )

    policy = RAGPolicy(
        allowed_collections=allowed_collections or None,
        denied_collections=denied_collections,
        max_retrievals_per_minute=max(0, _int_setting("AGT_MAX_RETRIEVALS_PER_MINUTE", 60)),
        rate_limit_window_seconds=max(1, _int_setting("AGT_RATE_LIMIT_WINDOW_SECONDS", 60)),
        content_policies=content_policies,
        audit_enabled=_bool_setting("AGT_AUDIT_ENABLED", True),
        audit_log_path=str(audit_path),
    )
    return GovernanceRuntime(
        governor=RAGGovernor(policy=policy, agent_id=agent_id),
        policy=policy,
        agent_id=agent_id,
        collection=collection,
        audit_path=audit_path,
        core_version=_package_version("agent-governance-toolkit-core"),
        rag_version=_package_version("agent-rag-governance"),
    )


def get_runtime() -> GovernanceRuntime:
    """Return the process-wide governance runtime, failing closed if absent."""

    global _RUNTIME, _RUNTIME_ERROR
    if _RUNTIME is not None:
        return _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            try:
                _RUNTIME = _create_runtime()
                _RUNTIME_ERROR = None
            except Exception as exc:
                _RUNTIME_ERROR = f"{type(exc).__name__}: {exc}"
                raise RuntimeError(_RUNTIME_ERROR) from exc
    return _RUNTIME


def _runtime_status() -> tuple[GovernanceRuntime | None, str | None]:
    try:
        return get_runtime(), None
    except RuntimeError as exc:
        return None, str(exc)


class _GovernedAsyncRetriever:
    """LangChain retriever adapter that cannot bypass AGT's sync guard."""

    def __init__(self, governed_retriever: Any) -> None:
        from langchain_core.retrievers import BaseRetriever
        from pydantic import PrivateAttr

        # BaseRetriever is a Pydantic model. Create a small dynamic subclass so
        # this module remains importable when optional LangChain dependencies
        # have not been installed yet (for example, during health checks).
        class Adapter(BaseRetriever):
            _governed: Any = PrivateAttr()

            def __init__(self, governed: Any, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self._governed = governed

            def _get_relevant_documents(self, query: str, *, run_manager: Any) -> list[Any]:
                return self._governed.invoke(query)

            async def _aget_relevant_documents(
                self, query: str, *, run_manager: Any
            ) -> list[Any]:
                return await run_in_threadpool(self._governed.invoke, query)

        self.adapter = Adapter(governed_retriever)


class GovernedVectorStoreProxy:
    """Delegate a Quivr vector store while governing every retriever it creates."""

    def __init__(self, vector_store: Any, runtime: GovernanceRuntime) -> None:
        self._vector_store = vector_store
        self._runtime = runtime

    def as_retriever(self, *args: Any, **kwargs: Any) -> Any:
        retriever = self._vector_store.as_retriever(*args, **kwargs)
        governed = self._runtime.governor.wrap(
            retriever, collection=self._runtime.collection
        )
        return _GovernedAsyncRetriever(governed).adapter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._vector_store, name)


def apply_governance(brain: Any) -> GovernanceRuntime:
    """Attach AGT to a Brain before Quivr can construct its RAG graph."""

    runtime = get_runtime()
    if not isinstance(getattr(brain, "vector_db", None), GovernedVectorStoreProxy):
        brain.vector_db = GovernedVectorStoreProxy(brain.vector_db, runtime)
    return runtime


def governance_snapshot(audit_limit: int = 40) -> dict[str, Any]:
    runtime, error = _runtime_status()
    if runtime is None:
        return {
            "enabled": False,
            "error": error,
            "agent_id": os.getenv("AGT_AGENT_ID", "mempalace-quivr"),
            "collection": os.getenv("AGT_COLLECTION", DEFAULT_COLLECTION),
            "capabilities": [],
            "audit_events": [],
        }

    policy = runtime.policy
    return {
        "enabled": True,
        "agent_id": runtime.agent_id,
        "collection": runtime.collection,
        "core_version": runtime.core_version,
        "rag_version": runtime.rag_version,
        "capabilities": runtime.capabilities,
        "policy": {
            "allowed_collections": getattr(policy, "allowed_collections", None),
            "denied_collections": getattr(policy, "denied_collections", []),
            "max_retrievals_per_minute": getattr(policy, "max_retrievals_per_minute", 0),
            "rate_limit_window_seconds": getattr(policy, "rate_limit_window_seconds", 60),
            "content_policies": getattr(policy, "content_policies", []),
            "audit_enabled": getattr(policy, "audit_enabled", False),
        },
        "audit_events": audit_events(audit_limit),
    }


def audit_events(limit: int = 40) -> list[dict[str, Any]]:
    runtime, _ = _runtime_status()
    if runtime is None or not runtime.audit_path.is_file():
        return []
    rows: deque[dict[str, Any]] = deque(maxlen=max(1, min(limit, 200)))
    try:
        with runtime.audit_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    rows.append(event)
    except OSError:
        return []
    return list(reversed(rows))


def evaluate_governance(collection: str, text: str) -> dict[str, Any]:
    """Dry-run the collection and content policies without logging a query."""

    runtime, error = _runtime_status()
    if runtime is None:
        return {"enabled": False, "decision": "unavailable", "reasons": [error]}

    collection = collection.strip() or runtime.collection
    reasons: list[str] = []
    collection_allowed, collection_reason = runtime.policy.is_collection_allowed(collection)
    if not collection_allowed:
        detail = f" ({collection_reason})" if collection_reason else ""
        reasons.append(f"Collection '{collection}' is not allowed by policy{detail}.")

    if text.strip():
        try:
            from agent_rag_governance import ContentScanner

            scan_results = ContentScanner(runtime.policy.content_policies).scan([text])
            blocked_result = next(
                (result for result in scan_results if getattr(result, "blocked", False)),
                None,
            )
            if blocked_result is not None:
                category = getattr(blocked_result, "category", None) or "content policy"
                reasons.append(f"Blocked by {category}.")
        except ImportError:
            reasons.append("Content scanner is unavailable in the installed toolkit.")

    return {
        "enabled": True,
        "decision": "deny" if reasons else "allow",
        "collection": collection,
        "text_chars": len(text),
        "reasons": reasons,
    }
