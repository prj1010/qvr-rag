"""Combined Quivr Core + MemPalace API serving the React web application."""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import os
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
import uvicorn

from quivr_core import Brain
from quivr_core.llm import LLMEndpoint
from quivr_core.rag.entities.config import DefaultModelSuppliers, LLMEndpointConfig
from quivr_mempalace import MemoryAwareAssistant, MempalaceMemoryStore, load_environment


APP_DIR = Path(__file__).resolve().parent
DEFAULT_PALACE = "~/.mempalace/palace"
LOCAL_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".mdx"}


@dataclass
class AppState:
    """Process-local state for the single-user local app and POC deployment."""

    brain: Brain | None = None
    indexed_files: list[str] = field(default_factory=list)
    indexed_chunks: int = 0
    kokoro: Any | None = None


STATE = AppState()

# Ensure Kokoro models exist, download if necessary
# On Render (Linux) use /tmp so it survives between requests in the same dyno.
# On Windows (local dev) put the models/ folder next to the app script.
_tmp_kokoro = Path("/tmp/kokoro_models")
MODELS_DIR = _tmp_kokoro if _tmp_kokoro.parent.exists() else APP_DIR / "models"
ONNX_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json"

def _ensure_kokoro_models() -> tuple[Path, Path]:
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    onnx_path = MODELS_DIR / "kokoro-v0_19.onnx"
    voices_path = MODELS_DIR / "voices.json"
    
    if not onnx_path.exists():
        print(f"Downloading Kokoro ONNX model to {onnx_path}...")
        urllib.request.urlretrieve(ONNX_URL, str(onnx_path))
    if not voices_path.exists():
        print(f"Downloading Kokoro voices to {voices_path}...")
        urllib.request.urlretrieve(VOICES_URL, str(voices_path))
        
    return onnx_path, voices_path

def get_kokoro():
    if STATE.kokoro is None:
        try:
            from kokoro_onnx import Kokoro
            onnx_path, voices_path = _ensure_kokoro_models()
            STATE.kokoro = Kokoro(str(onnx_path), str(voices_path))
        except ImportError:
            raise RuntimeError("kokoro-onnx is not installed.")
    return STATE.kokoro


def _ocr_pdf_documents(path: Path, metadata: dict[str, Any]) -> list[Document]:
    """OCR an image-only PDF with local PyMuPDF and Tesseract."""

    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "OCR requires PyMuPDF. Run setup.cmd, then install Tesseract."
        ) from exc

    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "OCR requires pytesseract and Pillow. Run setup.cmd, then install Tesseract."
        ) from exc

    tesseract_cmd = os.getenv("TESSERACT_CMD", "").strip()
    if not tesseract_cmd:
        default_tesseract_paths = [
            Path(os.getenv("ProgramFiles", r"C:\Program Files"))
            / "Tesseract-OCR"
            / "tesseract.exe",
            Path(os.getenv("ProgramFiles(x86)", r"C:\Program Files (x86)"))
            / "Tesseract-OCR"
            / "tesseract.exe",
            Path(os.getenv("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
            / "Programs"
            / "Tesseract-OCR"
            / "tesseract.exe",
        ]
        for candidate in default_tesseract_paths:
            if candidate.exists():
                tesseract_cmd = str(candidate)
                break
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise RuntimeError(
            "Tesseract OCR is not installed or not on PATH. Run "
            "install_ocr.cmd, or set TESSERACT_CMD in .env."
        ) from exc

    try:
        dpi = int(os.getenv("QUIVR_OCR_DPI", "150"))
    except ValueError as exc:
        raise ValueError("QUIVR_OCR_DPI must be an integer.") from exc
    if dpi < 72 or dpi > 400:
        raise ValueError("QUIVR_OCR_DPI must be between 72 and 400.")

    scale = dpi / 72
    documents: list[Document] = []
    with fitz.open(str(path)) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale),
                alpha=False,
            )
            image = Image.frombytes(
                "RGB",
                (pixmap.width, pixmap.height),
                pixmap.samples,
            )
            text = pytesseract.image_to_string(image)
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={**metadata, "page": page_number, "ocr": True},
                    )
                )

    return documents


def _load_local_documents(file_paths: list[str]) -> list[Document]:
    """Read common document types locally without Megaparse/NATS."""

    documents: list[Document] = []
    for raw_path in file_paths:
        path = Path(raw_path)
        suffix = path.suffix.lower()
        metadata = {
            "source": str(path),
            "original_file_name": path.name,
        }

        if suffix in LOCAL_TEXT_EXTENSIONS:
            text = path.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                documents.append(Document(page_content=text, metadata=metadata))
            continue

        if suffix == ".csv":
            with path.open(
                "r", encoding="utf-8", errors="replace", newline=""
            ) as handle:
                rows = csv.reader(handle)
                text = "\n".join(", ".join(row) for row in rows)
            if text.strip():
                documents.append(Document(page_content=text, metadata=metadata))
            continue

        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError(
                    "PDF support requires pypdf. Run setup.cmd again."
                ) from exc

            reader = PdfReader(str(path))
            pdf_documents: list[Document] = []
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pdf_documents.append(
                        Document(
                            page_content=text,
                            metadata={**metadata, "page": page_number},
                        )
                    )
            if pdf_documents:
                documents.extend(pdf_documents)
            else:
                documents.extend(_ocr_pdf_documents(path, metadata))
            continue

        if suffix == ".docx":
            try:
                from docx import Document as WordDocument
            except ImportError as exc:
                raise RuntimeError(
                    "DOCX support requires python-docx. Run setup.cmd again."
                ) from exc

            word_document = WordDocument(str(path))
            parts = [
                paragraph.text
                for paragraph in word_document.paragraphs
                if paragraph.text.strip()
            ]
            for table in word_document.tables:
                parts.extend(
                    ", ".join(cell.text for cell in row.cells)
                    for row in table.rows
                )
            text = "\n".join(parts)
            if text.strip():
                documents.append(Document(page_content=text, metadata=metadata))
            continue

        raise ValueError(
            f"Unsupported file type: {path.name}. Use PDF, DOCX, CSV, TXT, or Markdown."
        )

    if not documents:
        raise ValueError(
            "No text could be extracted. Scanned/image-only PDFs need OCR or Megaparse."
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
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} in the app .env file before continuing.")
    return value


def _build_llm(provider: str, model_name: str) -> LLMEndpoint:
    _load_app_environment()
    model_name = model_name.strip() or _default_model(provider)

    if provider == "NVIDIA NIM":
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
    else:
        config_key = _required("GROQ_API_KEY")
        chat_model = ChatGroq(
            model=model_name,
            api_key=config_key,
            temperature=0.2,
        )
        supplier = DefaultModelSuppliers.GROQ
        config_url = None

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


def _build_embedder() -> NVIDIAEmbeddings:
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
        chunks = _load_local_documents(file_paths)
        brain = asyncio.run(
            Brain.afrom_langchain_documents(
                name=brain_name.strip() or "mempalace-quivr",
                langchain_documents=chunks,
                llm=llm,
                embedder=_build_embedder(),
            )
        )
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
        system_prompt = (
            "Answer the question using the uploaded documents. Prior MemPalace "
            "memories are context only; ignore them if they are irrelevant.\n\n"
            f"{memory_context}"
        )
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
        memory_context = assistant.build_context(question)
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


app = FastAPI(
    title="Quivr + MemPalace RAG API",
    version="0.2.0",
    description="Document RAG and long-term memory API for the React frontend.",
)
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


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "brain_loaded": STATE.brain is not None,
        "indexed_files": len(STATE.indexed_files),
        "indexed_chunks": STATE.indexed_chunks,
    }


@app.post("/api/index")
def api_index_documents(
    files: list[UploadFile] = File(...),
    provider: str = Form("Groq"),
    model_name: str = Form(""),
    brain_name: str = Form("mempalace-quivr"),
) -> dict[str, Any]:
    """Accept browser uploads, parse them locally, and build the Quivr Brain."""

    if not files:
        raise HTTPException(status_code=400, detail="Select at least one document.")

    with tempfile.TemporaryDirectory(prefix="quivr-mempalace-") as temp_dir:
        temp_root = Path(temp_dir)
        saved_paths: list[str] = []
        original_names: list[str] = []
        for index, upload in enumerate(files, start=1):
            original_name = Path(upload.filename or f"upload-{index}.txt").name
            destination = temp_root / f"{index:03d}_{original_name}"
            destination.write_bytes(upload.file.read())
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

    STATE.brain = brain
    STATE.indexed_files = original_names
    STATE.indexed_chunks = _load_chunk_count(message)
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


@app.post("/api/recall")
def api_recall_memories(request: RecallRequest) -> dict[str, Any]:
    context = recall_memories(
        request.question,
        request.wing,
        request.palace_path,
        request.n_results,
    )
    return {"ok": not context.startswith("Memory retrieval error:"), "context": context}


@app.post("/api/ask")
async def api_answer_question(request: AskRequest) -> dict[str, Any]:
    if STATE.brain is None:
        return {
            "ok": False,
            "history": request.history,
            "context": "",
            "status": "Index documents first.",
        }

    history, _, context, status = await run_in_threadpool(
        answer_question,
        request.question,
        request.history,
        STATE.brain,
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
    text: str
    voice: str = "af_bella"

@app.post("/api/tts")
async def api_tts(request: TTSRequest):
    """Generate audio for the given text using Kokoro-ONNX."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    try:
        import soundfile as sf
        kokoro = get_kokoro()
        
        # Generate the audio samples
        samples, sample_rate = await run_in_threadpool(
            kokoro.create,
            request.text,
            voice=request.voice,
            speed=1.0,
            lang="en-us",
        )
        
        # Write to in-memory buffer
        buffer = io.BytesIO()
        sf.write(buffer, samples, sample_rate, format="WAV")
        buffer.seek(0)
        
        return StreamingResponse(buffer, media_type="audio/wav")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


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
