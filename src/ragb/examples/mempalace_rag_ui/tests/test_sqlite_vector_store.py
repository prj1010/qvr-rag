import asyncio
import unittest
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from sqlite_vector_store import SQLiteVecStore


class DeterministicEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            float("python" in lowered),
            float("sqlite" in lowered),
            float("render" in lowered),
        ]


class SQLiteVecStoreTests(unittest.TestCase):
    def test_add_search_delete_and_cleanup(self) -> None:
        path = Path.cwd() / "test-vectors.sqlite3"
        for suffix in ("", "-wal", "-shm"):
            path.with_name(path.name + suffix).unlink(missing_ok=True)
        self.addCleanup(
            lambda: [
                path.with_name(path.name + suffix).unlink(missing_ok=True)
                for suffix in ("", "-wal", "-shm")
            ]
        )

        store = SQLiteVecStore(path, DeterministicEmbeddings())
        documents = [
            Document(page_content="Python deployment", metadata={"source": "python"}),
            Document(page_content="SQLite vector storage", metadata={"source": "sqlite"}),
        ]
        ids = asyncio.run(
            store.aadd_documents(
                documents,
                ids=["python-doc", "sqlite-doc"],
            )
        )

        self.assertEqual(ids, ["python-doc", "sqlite-doc"])
        results = store.similarity_search_with_score("SQLite", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].id, "sqlite-doc")
        self.assertEqual(results[0][0].metadata["source"], "sqlite")

        self.assertTrue(store.delete(["sqlite-doc"]))
        self.assertEqual(store.similarity_search("SQLite", k=2)[0].id, "python-doc")
        store.cleanup()
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
