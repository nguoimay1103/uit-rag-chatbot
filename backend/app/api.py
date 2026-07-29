"""
api.py — FastAPI Production Backend
- JWT Auth via Supabase
- Chat history lưu vào Supabase PostgreSQL
- SSE Streaming
- Semantic Cache
"""
import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from fastapi import FastAPI, HTTPException, Depends, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import json
import asyncio
from typing import Optional

# ─── Imports nội bộ ─────────────────────────────────────────────────────────
try:
    from .chatbot import app as rag_agent, embeddings
except ImportError:
    try:
        from app.chatbot import app as rag_agent, embeddings
    except ImportError:
        from chatbot import app as rag_agent, embeddings

try:
    from .cache import SemanticCache
except ImportError:
    try:
        from app.cache import SemanticCache
    except ImportError:
        from cache import SemanticCache

try:
    from .auth import get_current_user
    from . import database as db
except ImportError:
    try:
        from app.auth import get_current_user
        from app import database as db
    except ImportError:
        from auth import get_current_user
        import database as db

# ─── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="UIT Học vụ RAG API",
    description="Chatbot tư vấn học vụ UIT — Production API",
    version="2.0.0",
)

# CORS — cho phép Vercel frontend và localhost dev
raw_origins = os.getenv("ALLOWED_ORIGINS", "")
env_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

default_origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://uit-rag-chatbot.vercel.app",
]

allowed_origins = list(set(env_origins + default_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Semantic Cache ──────────────────────────────────────────────────────────
semantic_cache = SemanticCache(embeddings_model=embeddings, threshold=0.95)

# ─── Pydantic Schemas ────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None      # UUID phiên chat (None = tạo mới)

class ChatResponse(BaseModel):
    answer: str
    session_id: str
    documents_retrieved: int
    confidence_score: float = 0.0
    cache_hit: bool = False

class CreateSessionRequest(BaseModel):
    title: str = "Cuộc trò chuyện mới"

class RenameSessionRequest(BaseModel):
    title: str


# ─── Root & Health Check ──────────────────────────────────────────────────────
@app.get("/")
@app.head("/")
async def root():
    return {
        "status": "ok",
        "service": "UIT Academic RAG API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
@app.head("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


# ─── Session Management ──────────────────────────────────────────────────────

@app.post("/api/v1/sessions")
async def create_session(
    req: CreateSessionRequest,
    user: dict = Depends(get_current_user),
):
    """Tạo phiên chat mới."""
    session = db.create_session(user["user_id"], req.title)
    return session


@app.get("/api/v1/sessions")
async def list_sessions(user: dict = Depends(get_current_user)):
    """Lấy danh sách phiên chat của user."""
    sessions = db.get_user_sessions(user["user_id"])
    return {"sessions": sessions}


@app.get("/api/v1/sessions/{session_id}/messages")
async def get_messages(
    session_id: str = Path(...),
    user: dict = Depends(get_current_user),
):
    """Lấy lịch sử tin nhắn của một phiên chat."""
    if not db.verify_session_owner(session_id, user["user_id"]):
        raise HTTPException(status_code=403, detail="Không có quyền truy cập phiên chat này.")
    messages = db.get_session_messages(session_id)
    return {"messages": messages}


@app.patch("/api/v1/sessions/{session_id}")
async def rename_session(
    req: RenameSessionRequest,
    session_id: str = Path(...),
    user: dict = Depends(get_current_user),
):
    """Đổi tên phiên chat."""
    updated = db.update_session_title(session_id, user["user_id"], req.title)
    return updated


@app.delete("/api/v1/sessions/{session_id}")
async def delete_session(
    session_id: str = Path(...),
    user: dict = Depends(get_current_user),
):
    """Xoá phiên chat."""
    if not db.verify_session_owner(session_id, user["user_id"]):
        raise HTTPException(status_code=403, detail="Không có quyền xoá phiên chat này.")
    db.delete_session(session_id, user["user_id"])
    return {"message": "Đã xoá phiên chat."}


# ─── Chat Endpoints ──────────────────────────────────────────────────────────

def _resolve_session(user_id: str, session_id: Optional[str]) -> str:
    """Lấy hoặc tạo session_id. Tạo mới nếu không có hoặc không hợp lệ."""
    if session_id and db.verify_session_owner(session_id, user_id):
        return session_id
    # Tạo phiên mới
    session = db.create_session(user_id)
    return session["id"]


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    user: dict = Depends(get_current_user),
):
    """Non-streaming chat endpoint (dự phòng)."""
    try:
        session_id = _resolve_session(user["user_id"], request.session_id)
        print(f"📥 [{user['email']}] [{session_id}] Câu hỏi: {request.question}")

        # Check cache
        cached = semantic_cache.lookup(request.question)
        if cached:
            db.save_message(session_id, "user", request.question)
            db.save_message(session_id, "assistant", cached["answer"],
                           confidence_score=1.0, docs_retrieved=cached["docs_count"])
            db.touch_session(session_id)
            return ChatResponse(
                answer=cached["answer"],
                session_id=session_id,
                documents_retrieved=cached["docs_count"],
                confidence_score=1.0,
                cache_hit=True,
            )

        # Load lịch sử từ DB
        history = db.get_recent_messages_for_context(session_id, last_n=6)

        config = {"configurable": {"thread_id": session_id}}
        result = rag_agent.invoke(
            {
                "question": request.question,
                "chat_history": history,
                "retry_count": 0,
                "confidence_score": 0.0,
            },
            config=config,
        )

        final_answer = result.get("answer", "Xin lỗi, hệ thống không thể trả lời câu hỏi này.")
        num_docs = len(result.get("documents", []))
        confidence = round(result.get("confidence_score", 0.0), 4)

        # Lưu vào DB
        db.save_message(session_id, "user", request.question)
        db.save_message(session_id, "assistant", final_answer,
                       confidence_score=confidence, docs_retrieved=num_docs)
        db.touch_session(session_id)

        # Cập nhật tiêu đề phiên nếu là tin nhắn đầu tiên
        messages = db.get_session_messages(session_id, limit=2)
        if len(messages) <= 2:
            db.update_session_title(session_id, user["user_id"], request.question)

        if num_docs > 0:
            semantic_cache.store(request.question, final_answer, num_docs)

        return ChatResponse(
            answer=final_answer,
            session_id=session_id,
            documents_retrieved=num_docs,
            confidence_score=confidence,
            cache_hit=False,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/chat/stream")
async def chat_stream_endpoint(
    request: ChatRequest,
    user: dict = Depends(get_current_user),
):
    """
    True Streaming Endpoint (Server-Sent Events).
    Bơm từng token từ LLM realtime tới client.
    """
    session_id = _resolve_session(user["user_id"], request.session_id)

    async def event_generator():
        try:
            # Gửi session_id về client ngay đầu để frontend lưu
            yield f"data: {json.dumps({'event': 'session', 'session_id': session_id}, ensure_ascii=False)}\n\n"

            # 1. Check Semantic Cache
            cached = semantic_cache.lookup(request.question)
            if cached:
                print(f"⚡ [Stream Cache HIT] [{user['email']}] [{session_id}]")
                # Lưu vào DB bất đồng bộ
                db.save_message(session_id, "user", request.question)

                chunk_size = 8
                ans = cached["answer"]
                for i in range(0, len(ans), chunk_size):
                    token = ans[i:i+chunk_size]
                    yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.01)

                db.save_message(session_id, "assistant", ans,
                               confidence_score=1.0, docs_retrieved=cached["docs_count"])
                db.touch_session(session_id)

                yield f"data: {json.dumps({'event': 'done', 'cache_hit': True, 'documents_retrieved': cached['docs_count'], 'confidence_score': 1.0, 'session_id': session_id}, ensure_ascii=False)}\n\n"
                return

            # 2. Load lịch sử từ DB
            history = db.get_recent_messages_for_context(session_id, last_n=6)

            # 3. Lưu câu hỏi của user ngay lập tức
            db.save_message(session_id, "user", request.question)

            # 4. Cập nhật tiêu đề nếu đây là tin nhắn đầu tiên
            existing_messages = db.get_session_messages(session_id, limit=2)
            if len(existing_messages) <= 1:
                db.update_session_title(session_id, user["user_id"], request.question)

            config = {"configurable": {"thread_id": session_id}}
            inputs = {
                "question": request.question,
                "chat_history": history,
                "retry_count": 0,
                "confidence_score": 0.0,
            }

            full_answer = ""
            confidence = 0.0
            num_docs = 0

            async for event in rag_agent.astream_events(inputs, config=config, version="v2"):
                event_type = event.get("event")
                tags = event.get("tags", [])

                # CHỈ stream token từ LLM chain trả lời (tag 'user_generation')
                if event_type == "on_chat_model_stream" and "user_generation" in tags:
                    chunk_data = event.get("data", {}).get("chunk")
                    if chunk_data and hasattr(chunk_data, "content") and chunk_data.content:
                        text_chunk = chunk_data.content
                        if isinstance(text_chunk, str):
                            full_answer += text_chunk
                            yield f"data: {json.dumps({'token': text_chunk}, ensure_ascii=False)}\n\n"

                elif event_type == "on_chain_end" and event.get("name") == "LangGraph":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict):
                        confidence = round(output.get("confidence_score", 0.0), 4)
                        num_docs = len(output.get("documents", []))
                        if not full_answer and "answer" in output:
                            full_answer = output["answer"]
                            yield f"data: {json.dumps({'token': full_answer}, ensure_ascii=False)}\n\n"

            # 5. Lưu câu trả lời vào DB
            if full_answer:
                db.save_message(session_id, "assistant", full_answer,
                               confidence_score=confidence, docs_retrieved=num_docs)
                db.touch_session(session_id)

            # 6. Cập nhật semantic cache
            if num_docs > 0 and full_answer:
                semantic_cache.store(request.question, full_answer, num_docs)

            yield f"data: {json.dumps({'event': 'done', 'cache_hit': False, 'documents_retrieved': num_docs, 'confidence_score': confidence, 'session_id': session_id}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Tắt Nginx buffering
        },
    )


# ─── User Info ───────────────────────────────────────────────────────────────

@app.get("/api/v1/me")
async def get_me(user: dict = Depends(get_current_user)):
    """Lấy thông tin user hiện tại."""
    return {"user_id": user["user_id"], "email": user["email"]}


# ─── Cache Management ────────────────────────────────────────────────────────

@app.get("/api/v1/cache/stats")
async def cache_stats(user: dict = Depends(get_current_user)):
    return {"cache_size": semantic_cache.size, "threshold": semantic_cache._threshold}


@app.delete("/api/v1/cache/clear")
async def cache_clear(user: dict = Depends(get_current_user)):
    semantic_cache.clear()
    return {"message": "Cache đã được xoá thành công."}


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)