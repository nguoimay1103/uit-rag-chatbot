"""
database.py — Supabase Database Client
Quản lý chat sessions và messages cho từng user.
Dùng supabase-py REST client (không cần SQLAlchemy).
"""
import os
from typing import Optional
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
# Dùng Service Role Key để bypass RLS ở phía server (key này KHÔNG chia sẻ với client)
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

_supabase_client: Optional[Client] = None


def get_supabase() -> Client:
    """Singleton Supabase client (service role — bypass RLS)."""
    global _supabase_client
    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            raise RuntimeError(
                "Thiếu SUPABASE_URL hoặc SUPABASE_SERVICE_ROLE_KEY trong environment variables."
            )
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _supabase_client


# ─── CHAT SESSIONS ──────────────────────────────────────────────────────────

def create_session(user_id: str, title: str = "Cuộc trò chuyện mới") -> dict:
    """Tạo phiên chat mới cho user."""
    db = get_supabase()
    result = (
        db.table("chat_sessions")
        .insert({"user_id": user_id, "title": title})
        .execute()
    )
    return result.data[0] if result.data else {}


def get_user_sessions(user_id: str, limit: int = 20) -> list:
    """Lấy danh sách phiên chat gần nhất của user."""
    db = get_supabase()
    result = (
        db.table("chat_sessions")
        .select("id, title, created_at, updated_at")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .limit(limit)
        .execute()
    )
    return result.data or []


def update_session_title(session_id: str, user_id: str, title: str) -> dict:
    """Cập nhật tiêu đề phiên chat (thường lấy từ câu hỏi đầu tiên)."""
    db = get_supabase()
    result = (
        db.table("chat_sessions")
        .update({"title": title[:80], "updated_at": "now()"})
        .eq("id", session_id)
        .eq("user_id", user_id)
        .execute()
    )
    return result.data[0] if result.data else {}


def touch_session(session_id: str) -> None:
    """Cập nhật updated_at của session khi có tin nhắn mới."""
    db = get_supabase()
    db.table("chat_sessions").update({"updated_at": "now()"}).eq("id", session_id).execute()


def delete_session(session_id: str, user_id: str) -> bool:
    """Xoá phiên chat (cascade xoá luôn messages)."""
    db = get_supabase()
    result = (
        db.table("chat_sessions")
        .delete()
        .eq("id", session_id)
        .eq("user_id", user_id)
        .execute()
    )
    return bool(result.data)


# ─── CHAT MESSAGES ──────────────────────────────────────────────────────────

def save_message(
    session_id: str,
    role: str,
    content: str,
    confidence_score: float = 0.0,
    docs_retrieved: int = 0,
) -> dict:
    """Lưu một tin nhắn vào database."""
    db = get_supabase()
    result = (
        db.table("chat_messages")
        .insert({
            "session_id": session_id,
            "role": role,
            "content": content,
            "confidence_score": confidence_score,
            "docs_retrieved": docs_retrieved,
        })
        .execute()
    )
    return result.data[0] if result.data else {}


def get_session_messages(session_id: str, limit: int = 50) -> list:
    """Lấy tối đa limit tin nhắn mới nhất, trả về theo thời gian tăng dần."""
    db = get_supabase()
    result = (
        db.table("chat_messages")
        .select("role, content, confidence_score, docs_retrieved, created_at")
        .eq("session_id", session_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return list(reversed(result.data or []))


def get_recent_messages_for_context(session_id: str, last_n: int = 6) -> list[dict]:
    """
    Lấy N tin nhắn gần nhất để đưa vào prompt context.
    Returns format: [{"role": "user"|"assistant", "content": "..."}]
    """
    messages = get_session_messages(session_id, limit=last_n)
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def verify_session_owner(session_id: str, user_id: str) -> bool:
    """Kiểm tra user có sở hữu session này không."""
    db = get_supabase()
    result = (
        db.table("chat_sessions")
        .select("id")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .execute()
    )
    return bool(result.data)
