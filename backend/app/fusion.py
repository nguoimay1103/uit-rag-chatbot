"""Các thuật toán fusion độc lập với LangChain để dễ kiểm thử."""

from collections.abc import Callable, Sequence
from typing import Any


def reciprocal_rank_fusion(
    ranked_lists: Sequence[tuple[Sequence[Any], float]],
    *,
    key_fn: Callable[[Any], str],
    rrf_k: int = 60,
) -> tuple[list[Any], list[float], float]:
    """Gộp nhiều danh sách xếp hạng bằng Weighted Reciprocal Rank Fusion.

    Mỗi phần tử của ranked_lists là (documents, weight). Một document xuất
    hiện trong nhiều retriever/query sẽ được cộng điểm, nhờ đó ưu tiên kết
    quả có sự đồng thuận thay vì phụ thuộc thang điểm riêng của BM25/dense.

    Hàm trả về documents, scores và normalized_top_score theo thứ tự giảm
    dần. normalized_top_score nằm trong [0, 1] và biểu diễn mức đồng thuận
    retrieval tương đối so với điểm tối đa có thể có ở rank 1.
    """
    if rrf_k <= 0:
        raise ValueError("rrf_k phải lớn hơn 0")

    scores: dict[str, float] = {}
    documents: dict[str, Any] = {}
    total_positive_weight = 0.0

    for ranked_documents, weight in ranked_lists:
        if weight <= 0:
            continue
        total_positive_weight += weight
        seen_in_list: set[str] = set()

        for rank, document in enumerate(ranked_documents, start=1):
            key = key_fn(document)
            if key in seen_in_list:
                continue
            seen_in_list.add(key)
            documents.setdefault(key, document)
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)

    ranked_keys = sorted(scores, key=scores.get, reverse=True)
    fused_documents = [documents[key] for key in ranked_keys]
    fused_scores = [scores[key] for key in ranked_keys]

    max_possible_score = total_positive_weight / (rrf_k + 1)
    normalized_top_score = (
        min(fused_scores[0] / max_possible_score, 1.0)
        if fused_scores and max_possible_score > 0
        else 0.0
    )
    return fused_documents, fused_scores, normalized_top_score
