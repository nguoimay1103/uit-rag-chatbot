import unittest
import io
from contextlib import redirect_stdout

from backend.app.cache import SemanticCache


class FakeEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        normalized = text.casefold()
        return [
            float(normalized.count("học")),
            float(normalized.count("bổng")),
            float(len(normalized) or 1),
        ]


class SemanticCacheIsolationTests(unittest.TestCase):
    def setUp(self):
        self.cache = SemanticCache(FakeEmbeddings(), threshold=0.99)
        with redirect_stdout(io.StringIO()):
            self.cache.store(
                "Điều kiện học bổng?",
                "Câu trả lời đã kiểm định",
                3,
                confidence_score=0.78,
                context_key="history-a",
                corpus_version="corpus-v1",
            )

    def test_hit_requires_same_context_and_corpus_version(self):
        with redirect_stdout(io.StringIO()):
            hit = self.cache.lookup(
                "Điều kiện học bổng?",
                context_key="history-a",
                corpus_version="corpus-v1",
            )

        self.assertIsNotNone(hit)
        self.assertEqual(hit["answer"], "Câu trả lời đã kiểm định")
        self.assertEqual(hit["confidence_score"], 0.78)

    def test_different_history_cannot_reuse_answer(self):
        self.assertIsNone(
            self.cache.lookup(
                "Điều kiện học bổng?",
                context_key="history-b",
                corpus_version="corpus-v1",
            )
        )

    def test_new_corpus_version_invalidates_old_answer(self):
        self.assertIsNone(
            self.cache.lookup(
                "Điều kiện học bổng?",
                context_key="history-a",
                corpus_version="corpus-v2",
            )
        )


if __name__ == "__main__":
    unittest.main()
