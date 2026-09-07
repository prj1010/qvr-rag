"""Document parsing, local Docling OCR, and optional cloud OCR fallback."""

from __future__ import annotations

from functools import lru_cache
import os
from threading import Lock
import time
from pathlib import Path


DOCLING_CONVERSION_LOCK = Lock()


class OlgaRequiresOcrError(RuntimeError):
    """Raised when Olga identifies a document that needs an OCR stage."""


def _open_olga_document(path: Path):
    """Open a document with Olga while keeping the dependency lazy."""

    try:
        import olgadoc
    except ImportError as exc:
        raise RuntimeError(
            "Olga document parsing is unavailable. Install olgadoc and restart."
        ) from exc

    try:
        return olgadoc.Document.open(str(path))
    except Exception as exc:
        raise RuntimeError(f"Olga could not open {path.name}: {exc}") from exc


def _olga_blocker_details(report: object) -> str:
    blockers = getattr(report, "blockers", [])
    details: list[str] = []
    for blocker in blockers:
        if isinstance(blocker, dict):
            details.append(str(blocker.get("kind", blocker)))
        else:
            details.append(str(getattr(blocker, "kind", blocker)))
    return ", ".join(details) or "native text is unavailable"


def extract_olga_text(path: Path) -> str:
    """Extract page-preserving Markdown with Olga for native documents.

    Olga intentionally does not OCR scanned-only PDFs. Those files raise
    ``OlgaRequiresOcrError`` so the caller can route them to Docling RapidOCR.
    """

    document = _open_olga_document(path)
    report = document.processability()
    is_blocked = getattr(report, "is_blocked", None)
    blocked = bool(is_blocked()) if callable(is_blocked) else not bool(
        getattr(document, "is_processable", True)
    )
    if blocked:
        message = (
            f"Olga found no native text in {path.name}; "
            f"({_olga_blocker_details(report)})."
        )
        if path.suffix.lower() == ".pdf":
            raise OlgaRequiresOcrError(f"{message} OCR is required.")
        raise RuntimeError(f"{message} An alternate parser is required.")

    try:
        page_markdown = document.markdown_by_page()
        if isinstance(page_markdown, dict):
            sections = [
                f"<!-- page {page_number} -->\n{content.strip()}"
                for page_number, content in sorted(page_markdown.items())
                if isinstance(content, str) and content.strip()
            ]
            markdown = "\n\n".join(sections)
        else:
            markdown = document.markdown()
    except Exception as exc:
        raise RuntimeError(f"Olga could not extract {path.name}: {exc}") from exc

    if isinstance(markdown, str) and markdown.strip():
        return markdown
    raise RuntimeError(f"Olga returned no text for {path.name}.")


def docstrange_fallback_enabled() -> bool:
    """Return whether the legacy cloud OCR fallback was explicitly enabled."""

    return os.getenv("DOCSTRANGE_FALLBACK_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@lru_cache(maxsize=1)
def _docling_converter():
    """Build one small, CPU-only Docling converter for scanned PDFs.

    Imports and model initialization stay lazy so normal text documents do not
    pay the Docling startup or memory cost.
    """

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.object_detection_engine_options import (
            OnnxRuntimeObjectDetectionEngineOptions,
        )
        from docling.datamodel.pipeline_options import (
            LayoutObjectDetectionOptions,
            OcrMode,
            PdfPipelineOptions,
            RapidOcrOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise RuntimeError(
            "Docling OCR dependencies are unavailable. Install "
            "docling-slim[format-pdf,feat-ocr-rapidocr-onnx] and restart. "
            f"Import error: {exc}"
        ) from exc

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=False,
        layout_options=LayoutObjectDetectionOptions.from_preset(
            "layout_heron_default",
            engine_options=OnnxRuntimeObjectDetectionEngineOptions(),
        ),
        ocr_options=RapidOcrOptions(
            lang=[os.getenv("DOCLING_OCR_LANG", "en").strip() or "en"],
            mode=OcrMode.FULL_PAGE,
        ),
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def extract_docling_text(path: Path) -> str:
    """Extract structured Markdown from a scanned PDF with local Docling OCR."""

    try:
        with DOCLING_CONVERSION_LOCK:
            conversion = _docling_converter().convert(path)
            markdown = conversion.document.export_to_markdown(
                traverse_pictures=True
            )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Docling OCR request failed: {exc}") from exc

    if isinstance(markdown, str) and markdown.strip():
        return markdown
    raise RuntimeError(f"Docling returned no text for {path.name}.")


def _docstrange_api_key() -> str:
    """Read the current and legacy Nanonets API-key environment names."""

    return (
        os.getenv("DOCSTRANGE_API_KEY", "").strip()
        or os.getenv("NANONETS_API_KEY", "").strip()
    )


def docstrange_configured() -> bool:
    """Return whether the DocStrange cloud API key is present."""

    return bool(_docstrange_api_key())


def extract_docstrange_text(path: Path) -> str:
    """Extract text from a document with DocStrange cloud OCR.

    Uses the provider's HTTP API directly so the app does not install the
    heavyweight local OCR/model dependencies from the DocStrange package.
    """

    api_key = _docstrange_api_key()
    if not api_key:
        raise RuntimeError(
            "Set DOCSTRANGE_API_KEY (or NANONETS_API_KEY) to OCR scanned "
            "documents with DocStrange."
        )

    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError(
            "DocStrange OCR requires the httpx package. Restart after running "
            "the current setup script."
        ) from exc

    api_url = os.getenv(
        "DOCSTRANGE_API_URL",
        "https://extraction-api.nanonets.com/api/v1/extract",
    ).strip()
    try:
        response = None
        for attempt in range(3):
            with path.open("rb") as document:
                response = httpx.post(
                    api_url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={
                        "file": (
                            path.name,
                            document,
                            "application/pdf",
                        )
                    },
                    data={"output_type": "markdown"},
                    timeout=httpx.Timeout(300.0, connect=30.0),
                )
            if response.status_code != 429 or attempt == 2:
                break

            retry_after = response.headers.get("retry-after", "")
            try:
                delay = max(1.0, min(float(retry_after), 15.0))
            except ValueError:
                delay = float(2**attempt)
            time.sleep(delay)

        assert response is not None
        if response.status_code == 429:
            detail = response.text.strip().replace("\n", " ")[:240]
            retry_after = response.headers.get("retry-after", "unknown")
            raise RuntimeError(
                "DocStrange rate limit reached after 3 attempts "
                f"(Retry-After: {retry_after}). {detail or 'Check the API plan and quota.'}"
            )
        response.raise_for_status()
        payload = response.json()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"DocStrange request failed: {exc}") from exc

    content = payload.get("content", "") if isinstance(payload, dict) else ""

    if isinstance(content, str) and content.strip():
        return content
    raise RuntimeError(f"DocStrange returned no text for {path.name}.")
