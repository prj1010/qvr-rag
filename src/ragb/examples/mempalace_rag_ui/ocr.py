"""Optional DocStrange cloud OCR integration for image-only documents."""

from __future__ import annotations

import os
from pathlib import Path


def docstrange_configured() -> bool:
    """Return whether the DocStrange cloud API key is present."""

    return bool(os.getenv("DOCSTRANGE_API_KEY", "").strip())


def extract_docstrange_text(path: Path) -> str:
    """Extract text from a document with DocStrange cloud OCR.

    Uses the provider's HTTP API directly so the app does not install the
    heavyweight local OCR/model dependencies from the DocStrange package.
    """

    api_key = os.getenv("DOCSTRANGE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Set DOCSTRANGE_API_KEY to OCR scanned documents with DocStrange."
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
        "https://extraction-api.nanonets.com/extract",
    ).strip()
    try:
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
        if response.status_code == 429:
            raise RuntimeError(
                "DocStrange rate limit reached. Check the API plan or try again later."
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
