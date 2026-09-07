"""Optional DocStrange cloud OCR integration for image-only documents."""

from __future__ import annotations

import os
import time
from pathlib import Path


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
        "https://extraction-api.nanonets.com/extract",
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
