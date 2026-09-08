import unittest

from backend.app.fusion import reciprocal_rank_fusion


class FakeDocument:
    def __init__(self, content: str):
        self.page_content = content


def document_key(document: FakeDocument) -> str:
    return document.page_content


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_consensus_document_beats_single_source_top_result(self):
        dense_only = FakeDocument("dense-only")
        bm25_only = FakeDocument("bm25-only")
        consensus = FakeDocument("consensus")

        documents, _, confidence = reciprocal_rank_fusion(
            [
                ([dense_only, consensus], 1.0),
                ([bm25_only, consensus], 1.0),
            ],
            key_fn=document_key,
        )

        self.assertEqual(documents[0].page_content, "consensus")
        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_duplicate_inside_one_list_is_counted_once(self):
        duplicate_a = FakeDocument("same")
        duplicate_b = FakeDocument("same")

        _, scores, _ = reciprocal_rank_fusion(
            [([duplicate_a, duplicate_b], 1.0)],
            key_fn=document_key,
            rrf_k=60,
        )

        self.assertEqual(len(scores), 1)
        self.assertAlmostEqual(scores[0], 1 / 61)

    def test_original_query_weight_can_break_a_tie(self):
        original_result = FakeDocument("original")
        rewritten_result = FakeDocument("rewritten")

        documents, _, _ = reciprocal_rank_fusion(
            [
                ([original_result], 1.25),
                ([rewritten_result], 1.0),
            ],
            key_fn=document_key,
        )

        self.assertEqual(documents[0].page_content, "original")


if __name__ == "__main__":
    unittest.main()
