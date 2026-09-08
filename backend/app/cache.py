"""
SemanticCache - Bộ nhớ đệm ngữ nghĩa cho hệ thống RAG UIT.

Cách hoạt động:
- Khi nhận câu hỏi mới, chỉ so sánh với entries cùng ngữ cảnh và phiên bản
  corpus, sau đó tính cosine similarity.
- Nếu similarity >= SIMILARITY_THRESHOLD -> trả về câu trả lời đã cache (bỏ qua pipeline).
- Nếu không khớp -> xử lý pipeline bình thường rồi lưu kết quả vào cache.

Lợi ích:
- Giảm 90%+ thời gian xử lý cho câu hỏi trùng/tương tự.
- Giảm chi phí gọi OpenAI API và Qdrant Cloud đáng kể.
"""

import time
import math
from typing import Optional

# ---- Cấu hình ----
SIMILARITY_THRESHOLD = 0.95   # Ngưỡng cosine similarity để coi là "cùng câu hỏi"
MAX_CACHE_SIZE = 200           # Số lượng câu hỏi tối đa lưu trong cache
CACHE_TTL_SECONDS = 3600       # Thời gian sống của cache entry (1 giờ)


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Tính cosine similarity giữa 2 vector embedding."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCache:
    """
    Bộ nhớ đệm ngữ nghĩa dùng cosine similarity trên embeddings.
    Thread-safe cho môi trường FastAPI asyncio đơn luồng.
    """

    def __init__(self, embeddings_model, threshold: float = SIMILARITY_THRESHOLD):
        self._embeddings = embeddings_model
        self._threshold = threshold
        # Mỗi entry gồm embedding, answer, context_key, corpus_version và metadata.
        self._store: list[dict] = []

    def _embed(self, text: str) -> list[float]:
        """Tính embedding cho một đoạn văn bản."""
        return self._embeddings.embed_query(text)

    def _evict_expired(self):
        """Xoá các entry đã hết TTL."""
        now = time.time()
        self._store = [e for e in self._store if now - e["timestamp"] < CACHE_TTL_SECONDS]

    def _evict_lru(self):
        """Nếu cache đầy, xoá entry cũ nhất (LRU đơn giản)."""
        if len(self._store) >= MAX_CACHE_SIZE:
            self._store.sort(key=lambda e: e["timestamp"])
            self._store.pop(0)

    def lookup(
        self,
        question: str,
        *,
        context_key: str = "root",
        corpus_version: str = "default",
    ) -> Optional[dict]:
        """
        Tìm câu trả lời đã cache cho câu hỏi.
        Trả về dict {"answer": ..., "docs_count": ...} nếu tìm thấy, None nếu cache miss.
        """
        self._evict_expired()
        if not self._store:
            return None

        candidates = [
            entry for entry in self._store
            if entry["context_key"] == context_key
            and entry["corpus_version"] == corpus_version
        ]
        if not candidates:
            return None

        q_embedding = self._embed(question)

        best_score = -1.0
        best_entry = None
        for entry in candidates:
            score = _cosine_similarity(q_embedding, entry["embedding"])
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self._threshold and best_entry is not None:
            print(f"⚡ [Cache HIT] Similarity={best_score:.4f} | Câu hỏi khớp: '{best_entry['question'][:60]}...'")
            # Cập nhật timestamp để "làm mới" entry (LRU touch)
            best_entry["timestamp"] = time.time()
            return {
                "answer": best_entry["answer"],
                "docs_count": best_entry["docs_count"],
                "confidence_score": best_entry["confidence_score"],
            }

        print(f"🔍 [Cache MISS] Best similarity={best_score:.4f} < threshold={self._threshold}")
        return None

    def store(
        self,
        question: str,
        answer: str,
        docs_count: int,
        *,
        confidence_score: float = 0.0,
        context_key: str = "root",
        corpus_version: str = "default",
    ):
        """Lưu câu hỏi + câu trả lời vào cache."""
        self._evict_expired()
        self._evict_lru()
        q_embedding = self._embed(question)
        self._store.append({
            "embedding": q_embedding,
            "question": question,
            "answer": answer,
            "docs_count": docs_count,
            "confidence_score": confidence_score,
            "context_key": context_key,
            "corpus_version": corpus_version,
            "timestamp": time.time(),
        })
        print(f"💾 [Cache STORE] Đã lưu câu hỏi vào cache. Kích thước cache hiện tại: {len(self._store)}")

    @property
    def size(self) -> int:
        return len(self._store)

    def clear(self):
        """Xoá toàn bộ cache (dùng khi cần reset)."""
        self._store.clear()
        print("🗑️ [Cache] Đã xoá toàn bộ cache.")
