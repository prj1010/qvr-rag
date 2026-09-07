import unittest

from model_catalog import MODEL_CATALOG, model_catalog_response


class ModelCatalogTests(unittest.TestCase):
    def test_catalog_exposes_supported_providers(self) -> None:
        self.assertEqual(
            set(MODEL_CATALOG), {"Groq", "NVIDIA NIM", "Microsoft Foundry"}
        )

    def test_catalog_includes_current_phi_and_slm_choices(self) -> None:
        nvidia_ids = {
            model["id"] for model in MODEL_CATALOG["NVIDIA NIM"]["models"]
        }
        foundry_ids = {
            model["id"] for model in MODEL_CATALOG["Microsoft Foundry"]["models"]
        }
        self.assertIn("microsoft/phi-4-mini-instruct", nvidia_ids)
        self.assertIn("nvidia/nemotron-nano-9b-v2", nvidia_ids)
        self.assertIn("Phi-4-mini-instruct", foundry_ids)
        self.assertIn("Phi-4-mini-reasoning", foundry_ids)

    def test_response_is_safe_to_serve_publicly(self) -> None:
        response = model_catalog_response()
        self.assertEqual(set(response), {"providers"})
        self.assertIs(response["providers"], MODEL_CATALOG)
        self.assertNotIn("api_key", str(response).lower())


if __name__ == "__main__":
    unittest.main()
