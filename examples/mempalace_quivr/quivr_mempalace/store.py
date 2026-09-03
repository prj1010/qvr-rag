"""The small storage boundary used by the memory-aware assistant.

MemPalace is imported lazily so the data model and assistant can be tested
without downloading ChromaDB or an embedding model. The concrete adapter uses
MemPalace's local collection and documented semantic-search API.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol


class MemoryStoreError(RuntimeError):
    """Raised when the MemPalace adapter cannot complete an operation."""


@dataclass(frozen=True)
class MemoryRecord:
    """Verbatim content and the MemPalace location where it belongs."""

    content: str
    wing: str
    room: str
    source_file: str = ""
    added_by: str = "quivr-mempalace"

    def __post_init__(self) -> None:
        for field_name in ("content", "wing", "room"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")

    def metadata(self) -> dict[str, str | int]:
        """Return metadata compatible with a MemPalace drawer."""

        return {
            "wing": self.wing,
            "room": self.room,
            "source_file": self.source_file,
            "added_by": self.added_by,
            "filed_at": datetime.now(timezone.utc).isoformat(),
            "chunk_index": 0,
        }


@dataclass(frozen=True)
class MemoryMatch:
    """A semantic-search hit returned by MemPalace."""

    content: str
    wing: str
    room: str
    similarity: float
    source_file: str = ""

    @classmethod
    def from_result(cls, result: dict[str, Any]) -> "MemoryMatch":
        return cls(
            content=str(result.get("text", "")),
            wing=str(result.get("wing", "")),
            room=str(result.get("room", "")),
            similarity=float(result.get("similarity", 0.0)),
            source_file=str(result.get("source_file", "")),
        )


class MemoryStore(Protocol):
    """Storage operations required by :class:`MemoryAwareAssistant`."""

    def add(self, memory: MemoryRecord) -> str:
        ...

    def search(
        self,
        query: str,
        *,
        wing: str | None = None,
        room: str | None = None,
        n_results: int = 5,
    ) -> list[MemoryMatch]:
        ...

    def wake_up(self, *, wing: str | None = None) -> str:
        ...


def stable_memory_id(memory: MemoryRecord) -> str:
    """Build an idempotent ID from the drawer's location and verbatim text."""

    identity = "\0".join((memory.wing, memory.room, memory.content))
    digest = sha256(identity.encode("utf-8")).hexdigest()
    return f"quivr-memory-{digest}"


class MempalaceMemoryStore:
    """Store and retrieve memories from a local MemPalace palace.

    The collection is opened only when the first operation is performed. This
    keeps importing the package cheap and makes missing optional runtime
    dependencies fail with a useful message at the point of use.
    """

    def __init__(
        self,
        palace_path: str | Path = "~/.mempalace/palace",
        *,
        wing: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.palace_path = Path(palace_path).expanduser().resolve()
        self.wing = wing
        self.collection_name = collection_name
        self._collection: Any | None = None

    def _load_collection(self) -> Any:
        if self._collection is not None:
            return self._collection

        try:
            from mempalace.config import MempalaceConfig
            from mempalace.palace import get_collection
        except ImportError as exc:
            raise MemoryStoreError(
                "MemPalace is not installed. Run `python -m pip install mempalace`."
            ) from exc

        try:
            collection_name = self.collection_name or MempalaceConfig().collection_name
            self._collection = get_collection(
                str(self.palace_path),
                collection_name=collection_name,
                create=True,
            )
        except Exception as exc:  # MemPalace has backend-specific exceptions.
            raise MemoryStoreError(
                f"Could not open MemPalace at {self.palace_path}: {exc}"
            ) from exc
        return self._collection

    def add(self, memory: MemoryRecord) -> str:
        """Upsert one verbatim memory and return its stable drawer ID."""

        drawer_id = stable_memory_id(memory)
        try:
            self._load_collection().upsert(
                ids=[drawer_id],
                documents=[memory.content],
                metadatas=[memory.metadata()],
            )
        except MemoryStoreError:
            raise
        except Exception as exc:
            raise MemoryStoreError(f"Could not write memory: {exc}") from exc
        return drawer_id

    def search(
        self,
        query: str,
        *,
        wing: str | None = None,
        room: str | None = None,
        n_results: int = 5,
    ) -> list[MemoryMatch]:
        """Run MemPalace semantic search and normalize the result shape."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if n_results < 1:
            raise ValueError("n_results must be at least 1")
        if not self.palace_path.exists():
            return []

        try:
            from mempalace.searcher import search_memories
        except ImportError as exc:
            raise MemoryStoreError(
                "MemPalace is not installed. Run `python -m pip install mempalace`."
            ) from exc

        try:
            payload = search_memories(
                query=query,
                palace_path=str(self.palace_path),
                wing=wing if wing is not None else self.wing,
                room=room,
                n_results=n_results,
                collection_name=self.collection_name,
            )
        except Exception as exc:  # MemPalace has backend-specific exceptions.
            raise MemoryStoreError(f"Could not search MemPalace: {exc}") from exc

        if payload.get("error"):
            raise MemoryStoreError(str(payload["error"]))
        return [MemoryMatch.from_result(item) for item in payload.get("results", [])]

    def wake_up(self, *, wing: str | None = None) -> str:
        """Load MemPalace's compact identity/story context for a session start."""

        if not self.palace_path.exists():
            return ""
        try:
            from mempalace.layers import MemoryStack
        except ImportError as exc:
            raise MemoryStoreError(
                "MemPalace is not installed. Run `python -m pip install mempalace`."
            ) from exc

        try:
            stack = MemoryStack(palace_path=str(self.palace_path))
            return stack.wake_up(wing=wing if wing is not None else self.wing)
        except Exception as exc:  # MemPalace has backend-specific exceptions.
            raise MemoryStoreError(f"Could not wake up from MemPalace: {exc}") from exc
