import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from ocr import (
    OlgaRequiresOcrError,
    docstrange_configured,
    docstrange_fallback_enabled,
    extract_docling_text,
    extract_docstrange_text,
    extract_olga_text,
)


class _FakeResponse:
    status_code = 200
    headers = {}
    text = ""

    def raise_for_status(self):
        pass

    def json(self):
        return {"content": "OCR extracted text"}


class _FakeHttpx:
    class Timeout:
        def __init__(self, *args, **kwargs):
            pass

    @staticmethod
    def post(url, headers, files, data, timeout):
        assert url == "https://extraction-api.nanonets.com/api/v1/extract"
        assert headers == {"Authorization": "Bearer test-key"}
        assert data == {"output_type": "markdown"}
        assert files["file"][0] == "test-ocr-scan.pdf"
        assert files["file"][2] == "application/pdf"
        assert files["file"][1].read() == b"pdf bytes"
        return _FakeResponse()


class _FakeDoclingDocument:
    def export_to_markdown(self, *, traverse_pictures):
        assert traverse_pictures is True
        return "Docling OCR extracted text"


class _FakeDoclingConversion:
    document = _FakeDoclingDocument()


class _FakeDoclingConverter:
    def convert(self, path):
        assert path.name == "test-ocr-scan.pdf"
        return _FakeDoclingConversion()


class _FakeOlgaReport:
    blockers = []

    def is_blocked(self):
        return False


class _FakeOlgaDocument:
    is_processable = True

    def processability(self):
        return _FakeOlgaReport()

    def markdown_by_page(self):
        return {2: "Second page", 1: "First page"}


class _FakeBlockedOlgaReport:
    blockers = [{"kind": "EmptyContent"}]

    def is_blocked(self):
        return True


class _FakeBlockedOlgaDocument:
    is_processable = False

    def processability(self):
        return _FakeBlockedOlgaReport()


class OcrTests(unittest.TestCase):
    def test_docstrange_requires_api_key(self) -> None:
        with patch.dict(
            os.environ,
            {"DOCSTRANGE_API_KEY": "", "NANONETS_API_KEY": ""},
            clear=False,
        ):
            self.assertFalse(docstrange_configured())

        with patch.dict(
            os.environ,
            {"DOCSTRANGE_API_KEY": "", "NANONETS_API_KEY": "legacy-key"},
            clear=False,
        ):
            self.assertTrue(docstrange_configured())

    def test_extract_docstrange_text_uses_cloud_extractor(self) -> None:
        path = Path.cwd() / "test-ocr-scan.pdf"
        self.addCleanup(path.unlink, missing_ok=True)
        with patch.dict(
            os.environ,
            {"DOCSTRANGE_API_KEY": "test-key"},
            clear=False,
        ), patch.dict(sys.modules, {"httpx": _FakeHttpx}):
            path.write_bytes(b"pdf bytes")
            self.assertEqual(extract_docstrange_text(path), "OCR extracted text")

    def test_extract_docling_text_uses_full_page_markdown(self) -> None:
        path = Path.cwd() / "test-ocr-scan.pdf"
        self.addCleanup(path.unlink, missing_ok=True)
        with patch("ocr._docling_converter", return_value=_FakeDoclingConverter()):
            path.write_bytes(b"pdf bytes")
            self.assertEqual(
                extract_docling_text(path), "Docling OCR extracted text"
            )

    def test_extract_olga_text_preserves_page_boundaries(self) -> None:
        path = Path.cwd() / "test-olga.pdf"
        with patch("ocr._open_olga_document", return_value=_FakeOlgaDocument()):
            self.assertEqual(
                extract_olga_text(path),
                "<!-- page 1 -->\nFirst page\n\n<!-- page 2 -->\nSecond page",
            )

    def test_extract_olga_text_routes_scanned_documents_to_ocr(self) -> None:
        path = Path.cwd() / "test-olga-scan.pdf"
        with patch(
            "ocr._open_olga_document", return_value=_FakeBlockedOlgaDocument()
        ):
            with self.assertRaisesRegex(OlgaRequiresOcrError, "EmptyContent"):
                extract_olga_text(path)

    def test_docstrange_fallback_is_opt_in(self) -> None:
        with patch.dict(os.environ, {"DOCSTRANGE_FALLBACK_ENABLED": "false"}):
            self.assertFalse(docstrange_fallback_enabled())
        with patch.dict(os.environ, {"DOCSTRANGE_FALLBACK_ENABLED": "true"}):
            self.assertTrue(docstrange_fallback_enabled())


if __name__ == "__main__":
    unittest.main()
