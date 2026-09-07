import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from ocr import docstrange_configured, extract_docstrange_text


class _FakeResponse:
    status_code = 200

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
        assert url == "https://extraction-api.nanonets.com/extract"
        assert headers == {"Authorization": "Bearer test-key"}
        assert data == {"output_type": "markdown"}
        assert files["file"][0] == "test-ocr-scan.pdf"
        assert files["file"][2] == "application/pdf"
        assert files["file"][1].read() == b"pdf bytes"
        return _FakeResponse()


class OcrTests(unittest.TestCase):
    def test_docstrange_requires_api_key(self) -> None:
        with patch.dict(
            os.environ,
            {"DOCSTRANGE_API_KEY": ""},
            clear=False,
        ):
            self.assertFalse(docstrange_configured())

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


if __name__ == "__main__":
    unittest.main()
