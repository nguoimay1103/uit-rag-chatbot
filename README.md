# UIT RAG Chatbot

[![CI](https://github.com/nguoimay1103/uit-rag-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/nguoimay1103/uit-rag-chatbot/actions/workflows/ci.yml)

Chatbot hỏi đáp học vụ cho Trường Đại học Công nghệ Thông tin (UIT), sử dụng
Retrieval-Augmented Generation để trả lời từ quy chế đào tạo và giảm câu trả lời
không có căn cứ.

## Kiến trúc

```text
Browser (Supabase Auth)
        |
        | JWT + SSE/JSON
        v
FastAPI API
        |
        v
LangGraph workflow
  Router -> Query rewrite -> Multi-query
        -> Qdrant dense search + BM25
        -> Weighted RRF / local BGE reranker
        -> Document grader -> Generate
        -> Hallucination verifier -> Retry / Safety fallback
        |
        +---- Supabase PostgreSQL (sessions and messages)
        +---- in-memory semantic cache
```

Production sử dụng `text-embedding-3-small`, Qdrant Cloud, BM25, Weighted
Reciprocal Rank Fusion và `gpt-4o-mini`. Local development có thể bật
`BAAI/bge-reranker-v2-m3` bằng `USE_RERANKER=true`.

## Thành phần chính

- `backend/app/api.py`: FastAPI, JWT, session API, verified SSE streaming và
  semantic cache.
- `backend/app/chatbot.py`: graph và toàn bộ RAG workflow.
- `backend/app/fusion.py`: Weighted Reciprocal Rank Fusion.
- `backend/app/database.py`: lưu session/history trên Supabase.
- `scripts/ingest.py`: semantic chunking, ingest Qdrant và tạo BM25 index.
- `scripts/evaluate.py`: system benchmark 72 case và regression gate.
- `frontend/index.html`: giao diện static, Supabase Auth và SSE client.

## Yêu cầu

- Python 3.11 được khuyến nghị.
- OpenAI API key.
- Qdrant Cloud cluster có collection `uit_admissions`.
- Supabase project với schema trong [DEPLOY.md](DEPLOY.md).
- Corpus riêng tại `data/raw/{train,val,test}.csv` nếu cần ingest hoặc chạy full
  benchmark.

## Chạy local

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend\requirements-prod.txt
Copy-Item .env.example .env
```

Điền credential vào `.env`, bảo đảm
`data/processed/bm25_retriever.pkl` tồn tại, rồi chạy:

```powershell
python -m uvicorn backend.app.api:app --reload --port 8000
python -m http.server 8501 --directory frontend
```

Mở `http://localhost:8501`. API docs ở `http://localhost:8000/docs` và health
check ở `http://localhost:8000/health`.

Có thể chạy cả hai service bằng Docker:

```powershell
docker compose up --build
```

## Ingest dữ liệu

`scripts/ingest.py` đọc ba CSV trong `data/raw`. Mỗi file cần các cột
`question`, `context`, `article`, và `document`. Script loại context trùng, dùng
semantic chunking, tạo lại collection Qdrant `uit_admissions`, rồi ghi BM25 index
vào `data/processed/bm25_retriever.pkl`.

Các dependency ingest/local reranker nằm trong `backend/requirements.txt`.

```powershell
pip install -r backend\requirements.txt
python scripts\ingest.py
```

Lưu ý: ingest dùng `force_recreate=True`, vì vậy collection hiện tại sẽ bị thay
thế. Tăng `RAG_CORPUS_VERSION` sau mỗi lần re-index để vô hiệu cache cũ.

## Kiểm thử và benchmark

```powershell
python -m unittest discover -s tests -v
python scripts\evaluate.py --profile smoke --production-mode --no-gate
python scripts\evaluate.py --profile full --production-mode
```

Benchmark full cần corpus private trong `data/raw`. Thiết kế metric, threshold và
chi phí được mô tả trong [BENCHMARK.md](BENCHMARK.md). Baseline hiện tại nằm tại
[docs/benchmark-baseline-v2.md](docs/benchmark-baseline-v2.md), cùng
[báo cáo public theo từng case](docs/benchmark-results/2026-09-08-full.md).

## Biến môi trường

Sao chép `.env.example` và cấu hình các nhóm biến:

- OpenAI: `OPENAI_API_KEY`.
- Qdrant: `QDRANT_URL`, `QDRANT_API_KEY`.
- Supabase: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`,
  `SUPABASE_JWT_SECRET`, `SUPABASE_ANON_KEY`.
- Runtime: `ALLOWED_ORIGINS`, `USE_RERANKER`, `BM25_PATH`,
  `RAG_CORPUS_VERSION`.

Không commit `.env`, raw corpus, service-role keys hoặc benchmark reports.

## Deploy

Thiết lập production Render + Vercel + Supabase được mô tả từng bước trong
[DEPLOY.md](DEPLOY.md).

## Đóng góp và bảo mật

Xem [CONTRIBUTING.md](CONTRIBUTING.md) trước khi mở pull request. Báo cáo lỗ hổng
theo [SECURITY.md](SECURITY.md), không đăng credential hoặc dữ liệu sinh viên vào
public issue.
