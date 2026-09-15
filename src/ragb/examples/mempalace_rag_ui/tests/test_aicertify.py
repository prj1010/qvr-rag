import os
import unittest
from unittest.mock import patch

import aicertify_integration


class AICertifyIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        aicertify_integration.clear_interactions()
        aicertify_integration._LAST_EVALUATION = None

    def test_capture_is_opt_in_and_bounded(self) -> None:
        with patch.dict(os.environ, {"AICERTIFY_CAPTURE_INTERACTIONS": "false"}):
            aicertify_integration.record_interaction("question", "answer")
        self.assertEqual(aicertify_integration.captured_interactions(), [])

        with patch.dict(os.environ, {"AICERTIFY_CAPTURE_INTERACTIONS": "true"}):
            for index in range(105):
                aicertify_integration.record_interaction(str(index), "answer")
        captured = aicertify_integration.captured_interactions()
        self.assertEqual(len(captured), 100)
        self.assertEqual(captured[0]["input_text"], "5")

    def test_evaluation_requires_captured_interactions(self) -> None:
        result = aicertify_integration.evaluate_aicertify()
        self.assertFalse(result["ok"])
        self.assertIn("No captured interactions", result["error"])

    def test_evaluation_uses_configured_external_worker(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AICERTIFY_CAPTURE_INTERACTIONS": "true",
                "AICERTIFY_PYTHON": "aicertify-python",
            },
        ):
            aicertify_integration.record_interaction("question", "answer")
            with patch(
                "aicertify_integration._evaluate_external",
                return_value={"eu_ai_act": {"report_path": "report.md"}},
            ) as evaluate:
                result = aicertify_integration.evaluate_aicertify()

        self.assertTrue(result["ok"])
        self.assertEqual(result["execution"], "external:aicertify-python")
        evaluate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
