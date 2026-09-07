"""Optional DocStrange cloud OCR integration for image-only documents."""

from __future__ import annotations

import os
from pathlib import Path


def docstrange_configured() -> bool:
    """Return whether the DocStrange cloud API key is present."""

    return bool(os.getenv("DOCSTRANGE_API_KEY", "").strip())


def extract_docstrange_text(path: Path) -> str:
    """Extract text from a document with DocStrange cloud OCR.

    DocStrange is imported only when this path is used so text-based
    documents do not pay the dependency or initialization cost.
    """

    api_key = os.getenv("DOCSTRANGE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Set DOCSTRANGE_API_KEY to OCR scanned documents with DocStrange."
        )

    try:
        from docstrange import DocumentExtractor
    except ImportError as exc:
        raise RuntimeError(
            "DocStrange OCR is configured but its package is not installed. "
            "Install docstrange and restart the app."
        ) from exc

    extractor = DocumentExtractor(api_key=api_key)
    result = extractor.extract(str(path))
    extract_markdown = getattr(result, "extract_markdown", None)
    content = extract_markdown() if callable(extract_markdown) else ""

    if isinstance(content, str) and content.strip():
        return content
    raise RuntimeError(f"DocStrange returned no text for {path.name}.")
