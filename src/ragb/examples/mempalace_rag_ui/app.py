"""Combined Quivr Core + MemPalace API serving the React web application."""

from __future__ import annotations

import argparse
import asyncio
import gc
import io
import os
import shutil
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
import uvicorn

from admin_auth import ADMIN_AUTH
from governance import apply_governance, evaluate_governance, governance_snapshot
from observability import OBSERVABILITY, content_metadata
from model_catalog import MODEL_CATALOG, model_catalog_response
from ocr import (
    OlgaRequiresOcrError,
    docstrange_configured,
    docstrange_fallback_enabled,
    extract_docling_text,
    extract_docstrange_text,
    extract_olga_text,
)
from prompts import build_rag_system_prompt
from quivr_core import Brain
from quivr_core.llm import LLMEndpoint
from quivr_core.rag.entities.config import DefaultModelSuppliers, LLMEndpointConfig
from quivr_mempalace import MemoryAwareAssistant, MempalaceMemoryStore, load_environment
from sqlite_vector_store import SQLiteVecStore


APP_DIR = Path(__file__).resolve().parent
DEFAULT_PALACE = "~/.mempalace/palace"
LOCAL_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".mdx"}
SUPPORTED_OLGA_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".html", ".htm"}
SUPPORTED_MARKITDOWN_EXTENSIONS = {".pdf", ".docx", ".csv", *LOCAL_TEXT_EXTENSIONS}
DEFAULT_MAX_UPLOAD_FILES = 10
DEFAULT_MAX_UPLOAD_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_UPLOAD_TOTAL_BYTES = 50 * 1024 * 1024
DEFAULT_MAX_PARSED_DOCUMENT_CHARS = 5_000_000
DEFAULT_MAX_PARSED_TOTAL_CHARS = 10_000_000


@dataclass
class AppState:
    """Process-local state for the single-user local app and POC deployment."""

    brain: Brain | None = None
    indexed_files: list[str] = field(default_factory=list)
    indexed_chunks: int = 0
    kokoro: Any | None = None


STATE = AppState()
BRAIN_LOCK = Lock()
KOKORO_INIT_LOCK = Lock()
KOKORO_SYNTHESIS_SEMAPHORE = asyncio.Semaphore(1)

# Ensure Kokoro models exist, downloading them on the first TTS request.
# Kokoro-ONNX 0.6.x uses the v1.0 model and a NumPy voices archive. Keep the
# cache configurable so local Windows runs can use the app directory while
# hosted Linux deployments can use an ephemeral writable directory.
DEFAULT_MODELS_DIR = APP_DIR / "models" if os.name == "nt" else Path("/tmp/kokoro_models")
MODELS_DIR = Path(
    os.getenv("KOKORO_MODELS_DIR", str(DEFAULT_MODELS_DIR))
).expanduser()
DEFAULT_ONNX_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/kokoro-v1.0.onnx"
)
DEFAULT_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/voices-v1.0.bin"
)
ONNX_URL = os.getenv("KOKORO_MODEL_URL", DEFAULT_ONNX_URL)
VOICES_URL = os.getenv("KOKORO_VOICES_URL", DEFAULT_VOICES_URL)
MODEL_FILENAME = os.getenv("KOKORO_MODEL_FILENAME", "kokoro-v1.0.onnx")
VOICES_FILENAME = os.getenv("KOKORO_VOICES_FILENAME", "voices-v1.0.bin")


def _download_kokoro_asset(url: str, destination: Path) -> None:
    """Download a Kokoro asset atomically so interrupted downloads are ignored."""

    temporary_path = destination.with_name(f"{destination.name}.download")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "quivr-mempalace-rag/0.2"},
    )
    last_error: Exception | None = None
    for attempt in range(3):
        print(f"Downloading Kokoro asset to {destination} (attempt {attempt + 1}/3)...")
        try:
            with urllib.request.urlopen(request, timeout=120) as response, temporary_path.open(
                "wb"
            ) as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            temporary_path.replace(destination)
            return
        except Exception as exc:
            last_error = exc
            temporary_path.unlink(missing_ok=True)
            if attempt < 2:
                time.sleep(2**attempt)

    raise RuntimeError(f"Could not download Kokoro asset from {url}: {last_error}") from last_error


def _ensure_kokoro_models() -> tuple[Path, Path]:
    global MODEL_FILENAME, MODELS_DIR, ONNX_URL, VOICES_FILENAME, VOICES_URL
    MODELS_DIR = Path(
        os.getenv("KOKORO_MODELS_DIR", str(DEFAULT_MODELS_DIR))
    ).expanduser()
    ONNX_URL = os.getenv("KOKORO_MODEL_URL", DEFAULT_ONNX_URL)
    VOICES_URL = os.getenv("KOKORO_VOICES_URL", DEFAULT_VOICES_URL)
    MODEL_FILENAME = os.getenv("KOKORO_MODEL_FILENAME", "kokoro-v1.0.onnx")
    VOICES_FILENAME = os.getenv("KOKORO_VOICES_FILENAME", "voices-v1.0.bin")
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    onnx_path = MODELS_DIR / MODEL_FILENAME
    voices_path = MODELS_DIR / VOICES_FILENAME

    if not onnx_path.is_file() or onnx_path.stat().st_size == 0:
        _download_kokoro_asset(ONNX_URL, onnx_path)
    if not voices_path.is_file() or voices_path.stat().st_size == 0:
        _download_kokoro_asset(VOICES_URL, voices_path)

    return onnx_path, voices_path

def get_kokoro():
    if STATE.kokoro is None:
        with KOKORO_INIT_LOCK:
            if STATE.kokoro is None:
                try:
                    from kokoro_onnx import Kokoro

                    onnx_path, voices_path = _ensure_kokoro_models()
                    STATE.kokoro = Kokoro(str(onnx_path), str(voices_path))
                except ImportError as exc:
                    raise RuntimeError("kokoro-onnx is not installed.") from exc
    return STATE.kokoro


@lru_cache(maxsize=1)
def _markitdown_converter() -> Any:
    try:
        from markitdown import MarkItDown
    except ImportError as exc:
        raise RuntimeError(
            "Document parsing requires MarkItDown. Run setup.cmd again."
        ) from exc
    return MarkItDown(enable_plugins=False)


def _load_markitdown_document(path: Path, metadata: dict[str, Any]) -> Document:
    """Convert a supported upload to compact Markdown without local OCR."""
    try:
        result = _markitdown_converter().convert(str(path))
    except Exception as exc:
        raise RuntimeError(
            f"Could not parse {path.name} with MarkItDown: {type(exc).__name__}: {exc}"
        ) from exc

    markdown = getattr(result, "markdown", None) or getattr(result, "text_content", "")
    parser_name = "markitdown"
    if not isinstance(markdown, str) or not markdown.strip():
        if path.suffix.lower() == ".pdf":
            docling_error: Exception | None = None
            try:
                markdown = extract_docling_text(path)
                parser_name = "docling-rapidocr"
            except Exception as exc:
                docling_error = exc

            if (
                (not isinstance(markdown, str) or not markdown.strip())
                and docstrange_fallback_enabled()
                and docstrange_configured()
            ):
                try:
                    markdown = extract_docstrange_text(path)
                    parser_name = "docstrange-cloud-fallback"
                except Exception as exc:
                    raise RuntimeError(
                        f"MarkItDown found no text in {path.name}; Docling OCR failed: "
                        f"{type(docling_error).__name__}: {docling_error}; "
                        f"DocStrange fallback failed: {type(exc).__name__}: {exc}"
                    ) from exc

            if not isinstance(markdown, str) or not markdown.strip():
                raise RuntimeError(
                    f"MarkItDown found no text in {path.name}; Docling OCR failed: "
                    f"{type(docling_error).__name__}: {docling_error}."
                ) from docling_error
        else:
            raise RuntimeError(
                f"No text could be extracted from {path.name} with MarkItDown."
            )
    return _document_from_markdown(path, metadata, markdown, parser_name)


def _document_from_markdown(
    path: Path,
    metadata: dict[str, Any],
    markdown: object,
    parser_name: str,
) -> Document:
    """Validate parser output and apply the shared document-size limit."""

    if not isinstance(markdown, str) or not markdown.strip():
        raise RuntimeError(f"No text could be extracted from {path.name}.")
    max_chars = _positive_int_setting(
        "MAX_PARSED_DOCUMENT_CHARS", DEFAULT_MAX_PARSED_DOCUMENT_CHARS
    )
    if len(markdown) > max_chars:
        raise RuntimeError(
            f"{path.name} produced {len(markdown)} characters; the configured "
            f"parsed-document limit is {max_chars}."
        )
    return Document(page_content=markdown, metadata={**metadata, "parser": parser_name})


def _load_olga_document(path: Path, metadata: dict[str, Any]) -> Document:
    """Parse a native-text document with Olga and preserve page boundaries."""

    try:
        markdown = extract_olga_text(path)
    except OlgaRequiresOcrError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Could not parse {path.name} with Olga: {type(exc).__name__}: {exc}"
        ) from exc
    return _document_from_markdown(path, metadata, markdown, "olga")


def _load_local_documents(file_paths: list[str]) -> list[Document]:
    """Read common document types locally without Megaparse/NATS."""

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    documents: list[Document] = []
    max_total_chars = _positive_int_setting(
        "MAX_PARSED_TOTAL_CHARS", DEFAULT_MAX_PARSED_TOTAL_CHARS
    )
    total_chars = 0
    for raw_path in file_paths:
        path = Path(raw_path)
        suffix = path.suffix.lower()
        metadata = {
            "source": str(path),
            "original_file_name": path.name,
        }

        if suffix in SUPPORTED_OLGA_EXTENSIONS:
            try:
                document = _load_olga_document(path, metadata)
            except OlgaRequiresOcrError:
                if suffix != ".pdf":
                    raise
                # Olga correctly identifies scanned PDFs but does not OCR them.
                # Reuse the existing PDF fallback chain, which tries MarkItDown,
                # then local Docling RapidOCR, then optional DocStrange.
                document = _load_markitdown_document(path, metadata)
            except Exception as olga_error:
                if suffix not in SUPPORTED_MARKITDOWN_EXTENSIONS:
                    raise
                try:
                    document = _load_markitdown_document(path, metadata)
                except Exception as fallback_error:
                    raise RuntimeError(
                        f"Olga failed for {path.name}: {olga_error}; "
                        f"MarkItDown fallback failed: {fallback_error}"
                    ) from fallback_error
        elif suffix in SUPPORTED_MARKITDOWN_EXTENSIONS:
            document = _load_markitdown_document(path, metadata)
        else:
            raise ValueError(
                f"Unsupported file type: {path.name}. Use PDF, DOCX, XLSX, HTML, "
                "CSV, TXT, or Markdown."
            )

        total_chars += len(document.page_content)
        if total_chars > max_total_chars:
            raise RuntimeError(
                "The uploaded documents exceed the configured parsed-text "
                f"limit of {max_total_chars} characters."
            )
        documents.append(document)

    if not documents:
        raise ValueError(
            "No text could be extracted. Scanned PDFs require Docling OCR."
        )

    try:
        chunk_size = int(os.getenv("QUIVR_CHUNK_SIZE", "1200"))
        chunk_overlap = int(os.getenv("QUIVR_CHUNK_OVERLAP", "150"))
    except ValueError as exc:
        raise ValueError(
            "QUIVR_CHUNK_SIZE and QUIVR_CHUNK_OVERLAP must be integers."
        ) from exc

    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("QUIVR_CHUNK_OVERLAP must be less than QUIVR_CHUNK_SIZE.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_index"] = index
    return chunks


def _load_app_environment() -> None:
    configured = os.getenv("QUIVR_MEMPALACE_ENV_FILE", "").strip()
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend([Path.cwd() / ".env", APP_DIR / ".env"])
    for env_file in candidates:
        if env_file.exists():
            load_environment(env_file)
            return
    load_environment(None)


def _default_model(provider: str) -> str:
    if provider == "NVIDIA NIM":
        return os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b")
    if provider == "Microsoft Foundry":
        return os.getenv("MICROSOFT_FOUNDRY_MODEL", "Phi-4-mini-instruct")
    return os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} in the app .env file before continuing.")
    return value


def _positive_int_setting(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _build_llm(provider: str, model_name: str) -> LLMEndpoint:
    _load_app_environment()
    model_name = model_name.strip() or _default_model(provider)

    if provider == "NVIDIA NIM":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        api_key = os.getenv("NVIDIA_API_KEY", "").strip()
        base_url = os.getenv(
            "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
        ).strip()
        if not api_key and "localhost" not in base_url and "127.0.0.1" not in base_url:
            raise RuntimeError(
                "Set NVIDIA_API_KEY for hosted NVIDIA NIM, or use a local "
                "NVIDIA_BASE_URL."
            )
        chat_model = ChatNVIDIA(
            model=model_name,
            nvidia_api_key=api_key or None,
            base_url=base_url or None,
            temperature=0.2,
        )
        # Older published Quivr Core releases do not expose NVIDIA in their
        # supplier enum, but they can still use a supplied ChatNVIDIA object.
        supplier = getattr(
            DefaultModelSuppliers,
            "NVIDIA",
            DefaultModelSuppliers.GROQ,
        )
        config_key = api_key
        config_url = base_url
    elif provider == "Microsoft Foundry":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Microsoft Foundry requires langchain-openai. Run setup.cmd again."
            ) from exc

        api_key = (
            os.getenv("MICROSOFT_FOUNDRY_API_KEY", "").strip()
            or os.getenv("OPENAI_API_KEY", "").strip()
        )
        base_url = (
            os.getenv("MICROSOFT_FOUNDRY_BASE_URL", "").strip()
            or os.getenv("OPENAI_BASE_URL", "").strip()
        )
        if not api_key:
            raise RuntimeError(
                "Set MICROSOFT_FOUNDRY_API_KEY for Microsoft Foundry inference."
            )
        if not base_url:
            raise RuntimeError(
                "Set MICROSOFT_FOUNDRY_BASE_URL to the Foundry OpenAI v1 endpoint."
            )
        chat_model = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            temperature=0.2,
        )
        supplier = getattr(
            DefaultModelSuppliers,
            "OPENAI",
            DefaultModelSuppliers.GROQ,
        )
        config_key = api_key
        config_url = base_url
    elif provider == "Groq":
        from langchain_groq import ChatGroq

        config_key = _required("GROQ_API_KEY")
        chat_model = ChatGroq(
            model=model_name,
            api_key=config_key,
            temperature=0.2,
        )
        supplier = DefaultModelSuppliers.GROQ
        config_url = None
    else:
        supported = ", ".join(MODEL_CATALOG)
        raise RuntimeError(
            f"Unsupported model provider '{provider}'. Choose one of: {supported}."
        )

    config = LLMEndpointConfig(
        supplier=supplier,
        model=model_name,
        llm_api_key=config_key or None,
        llm_base_url=config_url,
        max_context_tokens=20000,
        max_output_tokens=4096,
        temperature=0.2,
    )
    return LLMEndpoint(llm_config=config, llm=chat_model)


def _build_embedder() -> Any:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    _load_app_environment()
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    base_url = os.getenv(
        "NVIDIA_EMBEDDING_BASE_URL",
        os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    ).strip()
    if not api_key and "localhost" not in base_url and "127.0.0.1" not in base_url:
        raise RuntimeError(
            "Set NVIDIA_API_KEY for hosted NVIDIA embeddings, or use a local "
            "NVIDIA_EMBEDDING_BASE_URL."
        )
    return NVIDIAEmbeddings(
        model=os.getenv(
            "NVIDIA_EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b"
        ),
        nvidia_api_key=api_key or None,
        base_url=base_url or None,
        max_batch_size=int(os.getenv("NVIDIA_EMBEDDING_BATCH_SIZE", "50")),
    )


def _paths(files: list[Any] | None) -> list[str]:
    paths = []
    for item in files or []:
        if isinstance(item, str):
            paths.append(item)
            continue
        path = getattr(item, "path", None)
        if path:
            paths.append(str(path))
            continue
        if isinstance(item, dict) and item.get("path"):
            paths.append(str(item["path"]))
    return paths


def index_documents(
    files: list[Any] | None,
    provider: str,
    model_name: str,
    brain_name: str,
) -> tuple[Any | None, str]:
    """Build a Quivr Brain from locally parsed files using NVIDIA embeddings."""

    file_paths = _paths(files)
    if not file_paths:
        return None, "Select at least one document."

    try:
        llm = _build_llm(provider, model_name)
        with OBSERVABILITY.span(
            "documents.parse",
            "retriever",
            {"file_count": len(file_paths)},
        ):
            chunks = _load_local_documents(file_paths)
        embedder = _build_embedder()
        vector_path = Path(tempfile.gettempdir()) / f"quivr-vectors-{uuid4().hex}.sqlite3"
        vector_store: SQLiteVecStore | None = None
        with OBSERVABILITY.span(
            "embeddings.index",
            "embedding",
            {"chunk_count": len(chunks), "provider": provider},
        ):
            try:
                vector_store = SQLiteVecStore(vector_path, embedding=embedder)
                brain = asyncio.run(
                    Brain.afrom_langchain_documents(
                        name=brain_name.strip() or "mempalace-quivr",
                        langchain_documents=chunks,
                        llm=llm,
                        embedder=embedder,
                        vector_db=vector_store,
                    )
                )
                apply_governance(brain)
            except Exception:
                if vector_store is not None:
                    vector_store.cleanup()
                else:
                    for suffix in ("", "-wal", "-shm"):
                        vector_path.with_name(vector_path.name + suffix).unlink(
                            missing_ok=True
                        )
                raise
        return brain, (
            f"Indexed {len(file_paths)} document(s) into {len(chunks)} local chunks "
            "with Quivr RAG."
        )
    except Exception as exc:
        return None, f"Indexing error: {type(exc).__name__}: {exc}"


class AsyncQuivrResponder:
    """Call Quivr's async RAG API safely from an API worker thread."""

    def __init__(self, brain: Brain) -> None:
        self.brain = brain

    def __call__(self, question: str, memory_context: str) -> str:
        system_prompt = build_rag_system_prompt(memory_context)
        response = asyncio.run(
            self.brain.aask(
                run_id=uuid4(),
                question=question,
                system_prompt=system_prompt,
            )
        )
        answer = getattr(response, "answer", response)
        if not isinstance(answer, str):
            raise TypeError("Quivr response did not contain a string answer")
        return answer


def _format_memories(matches: list[Any]) -> str:
    if not matches:
        return "No relevant MemPalace memories found."
    lines = ["Relevant MemPalace memories:"]
    for index, match in enumerate(matches, start=1):
        lines.append(
            f"[{index}] {match.wing}/{match.room} | similarity={match.similarity:.3f}"
        )
        lines.append(match.content)
    return "\n".join(lines)


def recall_memories(
    question: str,
    wing: str,
    palace_path: str,
    n_results: int,
) -> str:
    if not question.strip():
        return "Enter a question to search memory."
    if not wing.strip():
        return "Enter a MemPalace wing."
    try:
        store = MempalaceMemoryStore(
            palace_path=palace_path.strip() or DEFAULT_PALACE,
            wing=wing.strip(),
        )
        with OBSERVABILITY.span(
            "memory.search",
            "retriever",
            {"wing": wing.strip(), "n_results": int(n_results)},
        ):
            return _format_memories(
                store.search(
                    question,
                    wing=wing.strip(),
                    n_results=int(n_results),
                )
            )
    except Exception as exc:
        return f"Memory retrieval error: {type(exc).__name__}: {exc}"


def answer_question(
    question: str,
    history: list[list[str]] | None,
    brain: Brain | None,
    wing: str,
    palace_path: str,
    n_results: int,
) -> tuple[list[list[str]], str, str, str]:
    history = history or []
    if not question.strip():
        return history, "", "Enter a question.", "Ready."
    if brain is None:
        return history, question, "Index documents first.", "No Quivr Brain loaded."
    if not wing.strip():
        return history, question, "Enter a MemPalace wing.", "Ready."

    try:
        store = MempalaceMemoryStore(
            palace_path=palace_path.strip() or DEFAULT_PALACE,
            wing=wing.strip(),
        )
        assistant = MemoryAwareAssistant(
            store=store,
            responder=AsyncQuivrResponder(brain),
            wing=wing.strip(),
            n_results=int(n_results),
        )
        with OBSERVABILITY.span(
            "memory.search",
            "retriever",
            {"wing": wing.strip(), "n_results": int(n_results)},
        ):
            memory_context = assistant.build_context(question)
        with OBSERVABILITY.span(
            "llm.answer",
            "generation",
            {"question_chars": len(question)},
        ):
            answer = assistant.ask(question)
        return (
            [*history, [question, answer]],
            "",
            memory_context,
            "Quivr RAG answered and the conversation was saved to MemPalace.",
        )
    except Exception as exc:
        return (
            history,
            question,
            f"RAG error: {type(exc).__name__}: {exc}",
            "Fix the configuration and try again.",
        )


class AskRequest(BaseModel):
    question: str
    history: list[list[str]] = Field(default_factory=list)
    wing: str = "quivr-demo"
    palace_path: str = DEFAULT_PALACE
    n_results: int = Field(default=5, ge=1, le=10)


class RecallRequest(BaseModel):
    question: str
    wing: str = "quivr-demo"
    palace_path: str = DEFAULT_PALACE
    n_results: int = Field(default=5, ge=1, le=10)


class GovernanceEvaluateRequest(BaseModel):
    collection: str = Field(default="", max_length=200)
    text: str = Field(default="", max_length=10000)


app = FastAPI(
    title="Quivr + MemPalace RAG API",
    version="0.2.0",
    description="Document RAG and long-term memory API for the React frontend.",
)
ADMIN_AUTH.install_session_middleware(app)
FRONTEND_DIST = APP_DIR / "frontend" / "dist"
PACKAGED_FRONTEND_DIST = APP_DIR / "quivr_mempalace_rag_ui" / "static"
if not (FRONTEND_DIST / "index.html").exists() and (
    PACKAGED_FRONTEND_DIST / "index.html"
).exists():
    FRONTEND_DIST = PACKAGED_FRONTEND_DIST

if (FRONTEND_DIST / "assets").is_dir():
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIST / "assets")),
        name="frontend-assets",
    )


@app.on_event("shutdown")
def flush_observability() -> None:
    OBSERVABILITY.flush()


@app.get("/admin/login", name="admin_login")
async def admin_login(request: Request) -> Response:
    """Start SSO, or redirect to the explicitly enabled temporary fallback."""

    return await ADMIN_AUTH.login(request)


@app.get("/admin/fallback", name="admin_fallback")
def admin_fallback() -> Response:
    """Render the temporary token login page when Microsoft SSO is unavailable."""

    return ADMIN_AUTH.fallback_page()


@app.post("/admin/fallback", name="admin_fallback_submit")
def admin_fallback_submit(
    request: Request, token: str = Form(...)
) -> Response:
    """Create the same protected admin session using the temporary token."""

    ADMIN_AUTH.create_fallback_session(request, token)
    return RedirectResponse(url="/?view=observability", status_code=303)


@app.get("/admin/callback", name="admin_callback")
async def admin_callback(request: Request) -> Response:
    """Validate the Microsoft identity and create an allowlisted admin session."""

    await ADMIN_AUTH.callback(request)
    return RedirectResponse(url="/?view=observability", status_code=303)


@app.get("/admin/logout", name="admin_logout")
def admin_logout(request: Request) -> Response:
    request.session.clear()
    return RedirectResponse(url="/", status_code=303)


@app.get("/api/admin/me")
def admin_me(request: Request) -> dict[str, Any]:
    identity = ADMIN_AUTH.current_admin(request)
    return {
        "authenticated": identity is not None,
        "configured": ADMIN_AUTH.configured,
        "fallback_configured": ADMIN_AUTH.fallback_configured,
        "auth_methods": [
            method
            for method, enabled in (
                ("microsoft", ADMIN_AUTH.configured),
                ("fallback", ADMIN_AUTH.fallback_configured),
            )
            if enabled
        ],
        "admin": identity,
    }


@app.get("/api/admin/observability")
def admin_observability(request: Request) -> dict[str, Any]:
    """Return recent safe trace metadata for the Microsoft-authenticated admin."""

    identity = ADMIN_AUTH.require_admin(request)
    return {
        "admin": identity,
        **OBSERVABILITY.snapshot(),
        "governance": governance_snapshot(),
    }


@app.get("/api/admin/governance")
def admin_governance(request: Request) -> dict[str, Any]:
    """Return the active Agent Governance Toolkit policy and audit metadata."""

    ADMIN_AUTH.require_admin(request)
    return governance_snapshot()


@app.post("/api/admin/governance/evaluate")
def admin_governance_evaluate(
    request: Request, payload: GovernanceEvaluateRequest
) -> dict[str, Any]:
    """Dry-run retrieval policy checks without executing a retrieval."""

    ADMIN_AUTH.require_admin(request)
    return evaluate_governance(payload.collection, payload.text)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "brain_loaded": STATE.brain is not None,
        "indexed_files": len(STATE.indexed_files),
        "indexed_chunks": STATE.indexed_chunks,
    }


@app.get("/api/models")
def models() -> dict[str, Any]:
    """Return the curated provider/model catalog used by the workspace UI."""

    return model_catalog_response()


@app.post("/api/index")
def api_index_documents(
    files: list[UploadFile] = File(...),
    provider: str = Form("Groq"),
    model_name: str = Form(""),
    brain_name: str = Form("mempalace-quivr"),
) -> dict[str, Any]:
    with OBSERVABILITY.start_trace(
        "rag.index",
        {
            "file_count": len(files),
            "provider": provider,
            "model": model_name or _default_model(provider),
            "brain_name": brain_name.strip() or "mempalace-quivr",
        },
    ) as trace:
        with trace.span(
            "rag.index_documents",
            "chain",
            {"file_count": len(files), "provider": provider},
        ):
            return _index_documents_route(files, provider, model_name, brain_name)


def _index_documents_route(
    files: list[UploadFile],
    provider: str,
    model_name: str,
    brain_name: str,
) -> dict[str, Any]:
    """Accept browser uploads, parse them locally, and build the Quivr Brain."""

    if not files:
        raise HTTPException(status_code=400, detail="Select at least one document.")
    try:
        max_files = _positive_int_setting("MAX_UPLOAD_FILES", DEFAULT_MAX_UPLOAD_FILES)
        max_file_bytes = _positive_int_setting(
            "MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES
        )
        max_total_bytes = _positive_int_setting(
            "MAX_UPLOAD_TOTAL_BYTES", DEFAULT_MAX_UPLOAD_TOTAL_BYTES
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if len(files) > max_files:
        raise HTTPException(
            status_code=413,
            detail=f"Upload at most {max_files} files per indexing request.",
        )

    with tempfile.TemporaryDirectory(prefix="quivr-mempalace-") as temp_dir:
        temp_root = Path(temp_dir)
        saved_paths: list[str] = []
        original_names: list[str] = []
        total_bytes = 0
        for index, upload in enumerate(files, start=1):
            original_name = Path(upload.filename or f"upload-{index}.txt").name
            destination = temp_root / f"{index:03d}_{original_name}"
            file_bytes = 0
            with destination.open("wb") as output:
                while chunk := upload.file.read(1024 * 1024):
                    file_bytes += len(chunk)
                    total_bytes += len(chunk)
                    if file_bytes > max_file_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                f"{original_name} exceeds the {max_file_bytes} byte "
                                "per-file upload limit."
                            ),
                        )
                    if total_bytes > max_total_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                f"The request exceeds the {max_total_bytes} byte "
                                "total upload limit."
                            ),
                        )
                    output.write(chunk)
            saved_paths.append(str(destination))
            original_names.append(original_name)

        brain, message = index_documents(
            saved_paths,
            provider,
            model_name,
            brain_name,
        )

    if brain is None:
        return {"ok": False, "message": message}

    # Swap only after the SQLite vector store is fully built. This avoids
    # exposing a half-built index to readers.
    with BRAIN_LOCK:
        old_brain = STATE.brain
        STATE.brain = brain
        if old_brain is not None:
            old_store = getattr(old_brain, "vector_db", None)
            old_store = getattr(old_store, "_vector_store", old_store)
            cleanup = getattr(old_store, "cleanup", None)
            if callable(cleanup):
                cleanup()
            else:
                close = getattr(old_store, "close", None)
                if callable(close):
                    close()
        STATE.indexed_files = original_names
        STATE.indexed_chunks = _load_chunk_count(message)
    gc.collect()
    return {
        "ok": True,
        "message": message,
        "files": original_names,
        "chunks": STATE.indexed_chunks,
    }


def _load_chunk_count(message: str) -> int:
    """Extract the chunk count from the stable index status message."""

    words = message.split()
    for index, word in enumerate(words):
        if word == "into" and index + 1 < len(words):
            try:
                return int(words[index + 1])
            except ValueError:
                break
    return 0


def _answer_with_current_brain(
    question: str,
    history: list[list[str]] | None,
    wing: str,
    palace_path: str,
    n_results: int,
) -> tuple[list[list[str]], str, str, str]:
    """Answer while holding the state lock so re-indexing cannot close the store."""
    with BRAIN_LOCK:
        return answer_question(
            question,
            history,
            STATE.brain,
            wing,
            palace_path,
            n_results,
        )


@app.post("/api/recall")
def api_recall_memories(request: RecallRequest) -> dict[str, Any]:
    with OBSERVABILITY.start_trace(
        "memory.recall",
        {
            "question_chars": len(request.question),
            "wing": request.wing.strip(),
            "n_results": request.n_results,
        },
    ) as trace:
        with trace.span("memory.retrieve", "retriever", {"n_results": request.n_results}):
            context = recall_memories(
                request.question,
                request.wing,
                request.palace_path,
                request.n_results,
            )
        return {"ok": not context.startswith("Memory retrieval error:"), "context": context}


@app.post("/api/ask")
async def api_answer_question(request: AskRequest) -> dict[str, Any]:
    with OBSERVABILITY.start_trace(
        "rag.ask",
        {
            **content_metadata(request.question, label="question"),
            "wing": request.wing.strip(),
            "n_results": request.n_results,
        },
    ) as trace:
        with BRAIN_LOCK:
            brain_loaded = STATE.brain is not None
        if not brain_loaded:
            return {
                "ok": False,
                "history": request.history,
                "context": "",
                "status": "Index documents first.",
            }

        with trace.span("rag.answer", "chain", {"history_turns": len(request.history)}):
            history, _, context, status = await run_in_threadpool(
                _answer_with_current_brain,
                request.question,
                request.history,
                request.wing,
                request.palace_path,
                request.n_results,
            )
        answer = history[-1][1] if history and len(history[-1]) > 1 else ""
        return {
            "ok": status.startswith("Quivr RAG answered"),
            "answer": answer,
            "history": history,
            "context": context,
            "status": status,
        }

class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    voice: str = Field(default="af_bella", min_length=1, max_length=64)

@app.post("/api/tts")
async def api_tts(request: TTSRequest):
    """Generate audio for the given text using Kokoro-ONNX."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    with OBSERVABILITY.start_trace(
        "tts.synthesis",
        {**content_metadata(request.text), "voice": request.voice},
    ) as trace:
        with trace.span("tts.generate", "generation", {"voice": request.voice}):
            try:
                import soundfile as sf

                async with KOKORO_SYNTHESIS_SEMAPHORE:
                    kokoro = await run_in_threadpool(get_kokoro)

                    samples, sample_rate = await run_in_threadpool(
                        kokoro.create,
                        request.text,
                        voice=request.voice,
                        speed=1.0,
                        lang="en-us",
                    )

                buffer = io.BytesIO()
                sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
                buffer.seek(0)

                return StreamingResponse(
                    buffer,
                    media_type="audio/wav",
                    headers={"Content-Disposition": "inline; filename=quivr-response.wav"},
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"TTS generation failed: {type(exc).__name__}: {exc}",
                ) from exc


@app.get("/{path:path}")
def frontend(path: str) -> Response:
    """Serve the compiled React app, while keeping `/api/*` JSON-only."""

    if path.startswith("api/"):
        return JSONResponse({"detail": "API route not found."}, status_code=404)

    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse(
        {
            "message": "React frontend is not built yet.",
            "next": "Run setup.cmd, then build_frontend.cmd, and start run_app.cmd again.",
        },
        status_code=503,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the combined Quivr RAG API")
    parser.add_argument(
        "--host",
        default=os.getenv("DATABRICKS_APP_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("DATABRICKS_APP_PORT", "7862")),
    )
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
