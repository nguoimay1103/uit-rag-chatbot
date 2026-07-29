"""
auth.py — Supabase JWT Authentication Middleware
Xác thực token từ Supabase Auth, bảo vệ các endpoint chat.
"""
import os
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Supabase JWT secret — lấy từ Supabase Dashboard > Settings > API > JWT Secret
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "").strip().strip("'\"")

security = HTTPBearer(auto_error=False)


def verify_supabase_token(token: str) -> dict:
    """Giải mã và xác thực JWT từ Supabase."""
    if not SUPABASE_JWT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server chưa cấu hình SUPABASE_JWT_SECRET trên Render."
        )
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
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token không hợp lệ ({str(e)}). Vui lòng đăng nhập lại.",
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
