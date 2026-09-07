import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from ocr import docstrange_configured, extract_docstrange_text


class _FakeResult:
    def extract_markdown(self):
        return "OCR extracted text"


class _FakeExtractor:
    def __init__(self, api_key):
        self.api_key = api_key

    def extract(self, path):
        assert path.endswith("test-ocr-scan.pdf")
        return _FakeResult()


class OcrTests(unittest.TestCase):
    def test_docstrange_requires_api_key(self) -> None:
        with patch.dict(
            os.environ,
            {"DOCSTRANGE_API_KEY": ""},
            clear=False,
        ):
            self.assertFalse(docstrange_configured())

    def test_extract_docstrange_text_uses_cloud_extractor(self) -> None:
        docstrange_module = types.ModuleType("docstrange")
        docstrange_module.DocumentExtractor = _FakeExtractor
        path = Path.cwd() / "test-ocr-scan.pdf"
        self.addCleanup(path.unlink, missing_ok=True)
        with patch.dict(
            os.environ,
            {"DOCSTRANGE_API_KEY": "test-key"},
            clear=False,
        ), patch.dict(sys.modules, {"docstrange": docstrange_module}):
            path.write_bytes(b"pdf bytes")
            self.assertEqual(extract_docstrange_text(path), "OCR extracted text")


if __name__ == "__main__":
    unittest.main()
