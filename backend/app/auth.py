"""
auth.py — Supabase JWT Authentication Middleware
Xác thực token từ Supabase Auth, bảo vệ các endpoint chat.
Hỗ trợ cả giải mã local (fast) và fallback qua Supabase Auth API (bulletproof).
"""
import os
import json
import urllib.request
import urllib.error
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", os.getenv("SUPABASE_ANON_KEY", ""))
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "").strip().strip("'\"")

security = HTTPBearer(auto_error=False)


def _verify_via_supabase_api(token: str) -> dict | None:
    """Xác thực trực tiếp qua REST API của Supabase Auth nếu local JWT verify thất bại."""
    if not SUPABASE_URL:
        return None
    url = f"{SUPABASE_URL}/auth/v1/user"
    headers = {
        "Authorization": f"Bearer {token}",
    }
    if SUPABASE_SERVICE_KEY:
        headers["apikey"] = SUPABASE_SERVICE_KEY

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                user_data = json.loads(resp.read().decode("utf-8"))
                user_id = user_data.get("id")
                email = user_data.get("email")
                if user_id:
                    return {"sub": user_id, "email": email, "raw": user_data}
    except Exception as e:
        print(f"⚠️ Supabase API verification fallback error: {e}")
        return None
    return None


def verify_supabase_token(token: str) -> dict:
    """Giải mã và xác thực JWT từ Supabase."""
    # 1. Thử giải mã local bằng SUPABASE_JWT_SECRET (nếu có)
    if SUPABASE_JWT_SECRET:
        try:
            payload = jwt.decode(
                token,
                SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                options={"verify_aud": False},
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token đã hết hạn. Vui lòng đăng nhập lại.",
            )
        except jwt.InvalidTokenError:
            # Thử tiếp phương án 2 (Supabase Auth API)
            pass

    # 2. Fallback: Hỏi trực tiếp Supabase Auth API (/auth/v1/user)
    api_user = _verify_via_supabase_api(token)
    if api_user:
        return api_user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token không hợp lệ hoặc đã hết hạn. Vui lòng đăng nhập lại.",
    )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """
    FastAPI Dependency: Trả về thông tin user từ JWT.
    Sử dụng: `user: dict = Depends(get_current_user)`
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Yêu cầu đăng nhập. Vui lòng cung cấp Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = verify_supabase_token(credentials.credentials)

    user_id = payload.get("sub")
    email = payload.get("email")

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token không chứa thông tin user.",
        )

    return {
        "user_id": user_id,
        "email": email,
        "raw": payload,
    }


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict | None:
    """
    FastAPI Dependency: Trả về user nếu có token, None nếu không.
    Dùng cho các endpoint public nhưng có thể personalize.
    """
    if credentials is None:
        return None
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None
