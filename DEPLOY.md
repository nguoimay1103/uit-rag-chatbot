# 🚀 Hướng Dẫn Deploy Production — UIT RAG Chatbot

Thời gian: ~90 phút | Chi phí: Miễn phí

## Kiến trúc sau khi deploy

```
https://your-app.vercel.app  (Frontend - Vercel Free)
         ↓ JWT Auth + API calls
https://uit-rag.onrender.com (Backend  - Render Free 750h/tháng)
         ↓                           ↓
Supabase PostgreSQL          Qdrant Cloud
(Auth + Chat History)        (Vector DB - đã có)
```

---

## BƯỚC 1: Tạo Project Supabase (15 phút)

### 1.1 — Tạo tài khoản & Project

1. Truy cập supabase.com → **Start your project**
2. Đăng nhập bằng GitHub
3. Click **New project**:
   - Name: `uit-rag-chatbot`
   - Region: **Southeast Asia (Singapore)**
4. Đợi ~2 phút để project khởi động

### 1.2 — Tạo Database Schema

Vào **SQL Editor** → **New query**, paste và chạy:

```sql
CREATE TABLE chat_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  title TEXT DEFAULT 'Cuộc trò chuyện mới',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE chat_messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content TEXT NOT NULL,
  confidence_score FLOAT DEFAULT 0,
  docs_retrieved INT DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_sessions_user_id ON chat_sessions(user_id);
CREATE INDEX idx_sessions_updated ON chat_sessions(updated_at DESC);
CREATE INDEX idx_messages_session_id ON chat_messages(session_id);

ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users own their sessions"
  ON chat_sessions FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users see messages in their sessions"
  ON chat_messages FOR ALL
  USING (session_id IN (
    SELECT id FROM chat_sessions WHERE user_id = auth.uid()
  ));
```

### 1.3 — Lấy API Keys

Vào **Settings → API**:

| Biến môi trường | Nơi lấy |
|---|---|
| `SUPABASE_URL` | Project URL (`https://xxx.supabase.co`) |
| `SUPABASE_ANON_KEY` | `anon public` key |
| `SUPABASE_SERVICE_ROLE_KEY` | `service_role secret` key |
| `SUPABASE_JWT_SECRET` | Settings → API → JWT Settings → JWT Secret |

### 1.4 — Bật Google OAuth (tùy chọn)

1. Vào **Authentication → Providers → Google**
2. Vào Google Cloud Console → Tạo OAuth 2.0 Client
3. Authorized redirect URIs: `https://YOUR_PROJECT.supabase.co/auth/v1/callback`
4. Dán Client ID và Secret vào Supabase

---

## BƯỚC 2: Deploy Backend lên Render.com (20 phút)

### 2.1 — Chuẩn bị & Push code

```bash
git add .
git commit -m "feat: production deploy with Supabase auth"
git push origin main
```

### 2.2 — Tạo Web Service

1. render.com → Đăng nhập GitHub → **New → Web Service**
2. Chọn repository của bạn
3. Cấu hình:
   - **Name**: `uit-rag-backend`
   - **Region**: Singapore
   - **Build Command**: `pip install -r backend/requirements-prod.txt`
   - **Start Command**: `python -m uvicorn backend.app.api:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free

### 2.3 — Environment Variables (Render Dashboard → Environment)

```
OPENAI_API_KEY            = sk-proj-...
QDRANT_URL                = https://xxx.aws.cloud.qdrant.io
QDRANT_API_KEY            = eyJhbGci...
SUPABASE_URL              = https://xxx.supabase.co
SUPABASE_ANON_KEY         = eyJhbGci...anon-public-key...
SUPABASE_SERVICE_ROLE_KEY = eyJhbGci...
SUPABASE_JWT_SECRET       = your-jwt-secret
USE_RERANKER              = false
RAG_CORPUS_VERSION        = uit_admissions-v1
BM25_PATH                 = data/processed/bm25_retriever.pkl
ALLOWED_ORIGINS           = https://your-app.vercel.app,http://localhost:8501
```

### 2.4 — Upload BM25 file

```bash
git add data/processed/bm25_retriever.pkl
git commit -m "feat: add BM25 data"
git push
```

### 2.5 — Kiểm tra

Sau deploy (~5 phút), truy cập:
`https://uit-rag-backend.onrender.com/health` → Phải thấy `{"status":"ok"}`

---

## BƯỚC 3: Deploy Frontend lên Vercel (10 phút)

### 3.1 — Cấu hình frontend

Frontend lấy `SUPABASE_URL` và `SUPABASE_ANON_KEY` từ endpoint browser-safe
`/api/v1/public-config`. Hai biến này phải được cấu hình trong Render; không
hardcode JWT vào `frontend/index.html`. Chỉ URL backend production được khai báo
trong frontend.

```bash
git add frontend/index.html
git commit -m "feat: set production URLs"
git push
```

### 3.2 — Deploy lên Vercel

**Cách 1 — CLI:**
```bash
npm i -g vercel
vercel --yes
# Khi hỏi Root Directory → nhập: frontend
```

**Cách 2 — Dashboard:**
vercel.com → New Project → Import GitHub → Root Directory: `frontend` → Deploy

### 3.3 — Cập nhật CORS & Supabase redirect

Render → Environment → `ALLOWED_ORIGINS = https://your-app.vercel.app`

Supabase → Authentication → URL Configuration:
- Site URL: `https://your-app.vercel.app`
- Redirect URLs: `https://your-app.vercel.app`

---

## BƯỚC 4: Kiểm tra End-to-End

- [ ] Mở `https://your-app.vercel.app` → Hiện màn hình Login
- [ ] Đăng ký tài khoản → nhận + xác nhận email → đăng nhập
- [ ] Đặt câu hỏi → nhận streaming response sạch
- [ ] Refresh trang → vẫn còn đăng nhập
- [ ] Sidebar hiện lịch sử chat
- [ ] Tạo cuộc hội thoại mới
- [ ] Đăng xuất → về Login

---

## Xử lý sự cố

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| "Failed to fetch" | Render đang sleep (15min idle) | Đợi 30s rồi thử lại |
| "401 Unauthorized" | JWT Secret sai | Kiểm tra Supabase → Settings → JWT Secret |
| "500 Internal Error" | BM25 path sai hoặc Qdrant lỗi | Xem Render Logs |
| Email xác nhận không đến | Supabase giới hạn 4 email/giờ | Dùng Google OAuth thay thế |

---

## Chi phí ước tính

| Dịch vụ | Giá | Giới hạn |
|---|---|---|
| Vercel | $0 | 100GB bandwidth/tháng |
| Render.com | $0 | 750h/tháng |
| Supabase | $0 | 500MB DB, 50k users |
| OpenAI API | ~$1-5/tháng | Tùy số lượt chat |

> Khi project lớn → nâng Render lên Starter ($7/tháng) để không cold start và bật lại BAAI Reranker.
