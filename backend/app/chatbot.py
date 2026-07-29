import os
import sys
import pickle
from typing import List, Any, Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
load_dotenv()

# ==========================================
# PHẦN 1: TÍCH HỢP CÔNG CỤ TÌM KIẾM (HYBRID)
# ==========================================
import math

# ── Adaptive Reranker Strategy ──────────────────────────────────────────────
# USE_RERANKER=true  (default, local dev)  → load BAAI/bge-reranker-v2-m3
# USE_RERANKER=false (production Render)   → Qdrant Hybrid RRF làm ranking
# ─────────────────────────────────────────────────────────────────────────────
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"

class LexicalReranker:
    """Reranker dự phòng thuần từ khóa — 0 RAM, 0 disk."""
    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            q_words = set(query.lower().split())
            t_words = text.lower().split()
            if not q_words or not t_words:
                scores.append(0.0)
                continue
            matches = sum(1 for w in t_words if w in q_words)
            jaccard = matches / (len(q_words) + len(set(t_words)) - matches + 1e-5)
            scores.append(float(jaccard * 5.0))
        return scores


class ResilientReranker:
    """Cố gắng tải BAAI, fallback về LexicalReranker nếu OOM/disk."""
    def __init__(self):
        try:
            print("Đang tải mô hình Reranker (BAAI/bge-reranker-v2-m3)...")
            encoder = HuggingFaceCrossEncoder(
                model_name="BAAI/bge-reranker-v2-m3",
                model_kwargs={"device": "cpu"}
            )
            self.client = encoder.client
            print("   ✅ Đã tải BAAI Reranker thành công.")
        except Exception as e:
            print(f"⚠️ Không đủ đĩa/RAM cho BAAI Reranker ({e}). Kích hoạt Lexical Reranker.")
            self.client = LexicalReranker()


if USE_RERANKER:
    _reranker_model = ResilientReranker()
else:
    print("ℹ️  [Adaptive Reranker] USE_RERANKER=false — Dùng Qdrant Hybrid RRF thay thế (production mode).")
    _reranker_model = None


def rerank_documents(query: str, documents: list) -> tuple[list, float]:
    """
    Xếp hạng tài liệu.
    - USE_RERANKER=true : BAAI cross-encoder (tốt nhất, ~1.5GB RAM)
    - USE_RERANKER=false: Qdrant Hybrid đã ranked → chỉ lấy top-5, tính confidence
    """
    if not documents:
        return [], 0.0

    if _reranker_model is not None:
        # ── Chế độ BAAI ──
        print(f"📊 [Reranker] BAAI Batch Mode — {len(documents)} tài liệu...")
        pairs = [[query, doc.page_content] for doc in documents]
        scores = _reranker_model.client.predict(pairs)
        scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in scored_docs[:5]]
        top_score = float(scored_docs[0][0]) if scored_docs else 0.0
        confidence = round(1 / (1 + math.exp(-top_score)), 4)
        print(f"   ✅ Top {len(top_docs)} | Confidence={confidence:.2%}")
        return top_docs, confidence
    else:
        # ── Chế độ Qdrant Hybrid RRF (production) ──
        # Qdrant đã sắp xếp theo RRF score, chỉ cần lấy top-5
        top_docs = documents[:5]
        # Ước tính confidence từ số lượng docs được retrieve
        confidence = round(min(0.5 + len(top_docs) * 0.08, 0.92), 4)
        print(f"📊 [Reranker] Qdrant RRF mode — Lấy top {len(top_docs)} | Confidence={confidence:.2%}")
        return top_docs, confidence
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
qdrant_vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="uit_admissions",
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)
qdrant_retriever = qdrant_vectorstore.as_retriever(search_kwargs={"k": 5})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BM25_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../data/processed/bm25_retriever.pkl"))
BM25_PATH = os.getenv("BM25_PATH", DEFAULT_BM25_PATH)

if not os.path.exists(BM25_PATH):
    candidates = [
        os.path.join(BASE_DIR, "bm25_retriever.pkl"),
        "data/processed/bm25_retriever.pkl",
        "bm25_retriever.pkl",
    ]
    for c in candidates:
        if os.path.exists(c):
            BM25_PATH = c
            break

print(f"Đang tải bộ tìm kiếm từ khóa BM25 từ {BM25_PATH}...")
with open(BM25_PATH, "rb") as f:
    bm25_retriever = pickle.load(f)

def custom_hybrid_search(query):
    qdrant_docs = qdrant_retriever.invoke(query)
    bm25_docs = bm25_retriever.invoke(query)
    
    unique_docs = []
    seen_content = set()
    for i in range(max(len(qdrant_docs), len(bm25_docs))):
        if i < len(bm25_docs) and bm25_docs[i].page_content not in seen_content:
            unique_docs.append(bm25_docs[i])
            seen_content.add(bm25_docs[i].page_content)
        if i < len(qdrant_docs) and qdrant_docs[i].page_content not in seen_content:
            unique_docs.append(qdrant_docs[i])
            seen_content.add(qdrant_docs[i].page_content)
    return unique_docs[:5]

# ==========================================
# PHẦN 2: KHỞI TẠO LLM VÀ GIÁM KHẢO (GRADER)
# ==========================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Cấu trúc ép LLM trả lời Yes/No
class GradeDocuments(BaseModel):
    """Đánh giá sự liên quan của tài liệu."""
    binary_score: str = Field(description="Tài liệu có liên quan không? Chỉ trả lời 'yes' hoặc 'no'.")

structured_llm_grader = llm.with_structured_output(GradeDocuments)

system_grader_prompt = """Bạn là chuyên gia chấm điểm tài liệu pháp lý tài ba của trường Đại học CNTT (UIT).
Nhiệm vụ của bạn là đánh giá xem đoạn tài liệu (context) dưới đây có chứa thông tin hữu ích hoặc bổ trợ nào giúp trả lời câu hỏi của sinh viên hay không.

⚠️ QUY TẮC CHẤM ĐIỂM BẮT BUỘC (TRÁNH QUÁ KHẮT KHE):
1. KHÔNG đòi hỏi đoạn văn bản phải chứa câu trả lời đầy đủ hoàn chỉnh 100%.
2. Chỉ cần văn bản có đề cập đến cùng chủ đề, từ khóa quan trọng, hoặc các điều khoản liên quan trực tiếp/gián tiếp đến câu hỏi là phải chấm 'yes'.
3. LINH HOẠT ĐỊNH LƯỢNG: Nếu sinh viên hỏi về số lượng cụ thể ("bao nhiêu tiền", "mấy ngày", "mấy tín chỉ") nhưng tài liệu chỉ ghi công thức tính, tỷ lệ phần trăm (%), hoặc quy định chung chung do Hiệu trưởng quyết định, thì VẪN PHẢI CHẤM 'yes'.
4. Chỉ chấm 'no' nếu tài liệu hoàn toàn lạc đề, nói về một quy chế khác hoàn toàn không có giá trị tham khảo cho câu hỏi.

Tài liệu: \n{context}\n\nCâu hỏi: {question}

Hãy đưa ra điểm số 'yes' (có liên quan) hoặc 'no' (không liên quan):"""
grade_prompt = ChatPromptTemplate.from_template(system_grader_prompt)
retrieval_grader = grade_prompt | structured_llm_grader

# --- GIÁM THỊ KIỂM TRA ẢO GIÁC (HALLUCINATION GRADER) ---
class GradeHallucinations(BaseModel):
    """Đánh giá xem câu trả lời có bịa đặt ngoài tài liệu hay không."""
    reasoning: str = Field(description="Giải thích ngắn gọn (1-2 câu) lý do tại sao bạn đưa ra quyết định này.")
    binary_score: str = Field(description="Câu trả lời có mâu thuẫn hoặc bịa đặt thêm so với tài liệu không? Trả lời 'yes' (ĐẠT) hoặc 'no' (BỊA ĐẶT).")

structured_llm_hallucination_grader = llm.with_structured_output(GradeHallucinations)

system_hallucination_prompt = """Bạn là một Giám thị kiểm định Ảo giác (Hallucination Grader) khách quan của trường UIT.
Nhiệm vụ của bạn là đánh giá xem câu trả lời (generation) của AI có được căn cứ dựa trên các tài liệu trích xuất (context) hay không.

QUY TẮC CHẤM ĐIỂM LINH HOẠT (TRÁNH QUÁ KHẮT KHE):
1. Chấm 'yes' (Không ảo giác) nếu các con số, điều kiện dùng để TRẢ LỜI CHO CÂU HỎI HIỆN TẠI đều tìm thấy trong tài liệu.
2. CHẤP NHẬN AI mở rộng từ viết tắt và sử dụng từ nối tự nhiên.
3. BỎ QUA các thông tin AI nhắc lại từ cuộc trò chuyện trước đó để tạo sự liên kết. Chỉ bắt lỗi nếu AI tự ý bịa ra một quy chế mới tinh cho câu hỏi hiện tại.

Câu hỏi hiện tại:
{question}

Tài liệu gốc (Context):
\n{documents}\n

Câu trả lời của AI (Generation):
\n{generation}\n

Hãy trả về nhãn 'yes' nếu câu trả lời trung thực với tài liệu, hoặc 'no' nếu câu trả lời bịa đặt:"""

hallucination_prompt = ChatPromptTemplate.from_template(system_hallucination_prompt)
hallucination_grader = hallucination_prompt | structured_llm_hallucination_grader
# ==========================================
# PHẦN 3: ĐỊNH NGHĨA STATE & CÁC NODE LANGGRAPH
# ==========================================
class GraphState(TypedDict):
    question: str
    standalone_question: str
    documents: List[Any]
    answer: str
    chat_history: List[dict]
    retry_count: int          # Đếm số lần tái sinh khi phát hiện ảo giác
    confidence_score: float   # Điểm tin cậy từ Reranker top-1 (0.0 - 1.0)
    is_web_searched: bool     # Flag cho CRAG Web Search
    needs_clarification: bool # Flag cho Self-RAG Ambiguity
    clarification_question: str
def reformulate_node(state: GraphState):
    question = state["question"]
    history = state.get("chat_history", [])

    # Nếu không có lịch sử (câu hỏi đầu tiên), dùng luôn câu hỏi gốc
    if not history:
        return {"standalone_question": question}

    # Chỉ lấy 4 tin nhắn gần nhất (2 lượt thoại) để tránh LLM bị nhiễu thông tin cũ
    recent_history = history[-4:]
    history_str = "\n".join(
        f"{'Sinh viên' if m['role'] == 'user' else 'Bot'}: {m['content']}"
        for m in recent_history
    )

    prompt = f"""Bạn là một chuyên gia ngôn ngữ học. Nhiệm vụ của bạn là đọc Lịch sử trò chuyện và Câu hỏi nối tiếp của người dùng.
    
    1. Nếu câu hỏi nối tiếp là một câu hỏi về quy chế bị khuyết chủ ngữ, hãy TÁI TẠO nó thành một câu hỏi ĐỘC LẬP đầy đủ ý nghĩa.
    2. Nếu câu hỏi CHỈ LÀ câu chào hỏi, cảm ơn, hoặc giao tiếp thông thường (VD: "chào bạn", "cảm ơn", "bot ngu ngốc"), hãy GIỮ NGUYÊN câu gốc, TUYỆT ĐỐI KHÔNG ghép nối với lịch sử quy chế.

    🌟 VÍ DỤ:
    - Lịch sử: Sinh viên: Học bổng xuất sắc được bao nhiêu? -> Bot: Được 120% học phí.
      Câu hỏi nối tiếp: Vậy loại giỏi thì sao?
      -> Câu tái tạo: Học bổng loại giỏi được bao nhiêu tiền?
    - Lịch sử: Sinh viên: Nợ môn có được làm Đồ án không? -> Bot: Không được nợ quá 8 tín chỉ.
      Câu hỏi nối tiếp: Chào bạn
      -> Câu tái tạo: Chào bạn

    Lịch sử cuộc trò chuyện:
    {history_str}

    Câu hỏi nối tiếp: {question}
    Trả lời (chỉ in ra câu được tái tạo, không giải thích):"""

    response = llm.invoke(prompt)
    standalone_q = response.content.strip()
    
    print(f"🔄 [Reformulate] Viết lại câu hỏi: '{question}' ➡️ '{standalone_q}'")
    return {"standalone_question": standalone_q}
def multi_query_generator(original_query):
    print("🧠 [Multi-Query] Đang nội suy các góc nhìn khác của câu hỏi...")
    
    # Ép LLM trả về danh sách phân tách bằng dấu phẩy
    output_parser = CommaSeparatedListOutputParser()
    
    prompt = PromptTemplate(
        template="""Bạn là một chuyên gia pháp chế đại học. 
        Nhiệm vụ của bạn là tạo ra 3 phiên bản diễn đạt khác nhau cho câu hỏi của sinh viên để tối ưu hóa việc tìm kiếm trong cơ sở dữ liệu Vector.
        
        ⚠️ QUY TẮC CHUẨN HÓA TỪ VỰNG (QUAN TRỌNG):
        Hãy chủ động dịch các từ ngữ "lóng", "đời thường" sang "ngôn ngữ học vụ/hành chính" chuẩn xác để máy tìm kiếm dễ hiểu:
        - "bao nhiêu tiền", "được mấy đồng" -> "mức học bổng", "quy định tài chính", "định mức học phí", "tỷ lệ phần trăm"
        - "rớt môn", "tạch" -> "điểm F", "không đạt", "học lại"
        - "đuổi học", "bị kick" -> "buộc thôi học", "cảnh cáo học vụ"
        
        Câu hỏi gốc: {query}
        \n{format_instructions}""",
        input_variables=["query"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()}
    )
    
    chain = prompt | llm | output_parser
    variants = chain.invoke({"query": original_query})
    
    # Trả về danh sách gồm câu gốc + 3 câu nội suy
    return [original_query] + variants
# --- BỘ ĐỊNH TUYẾN (SEMANTIC ROUTER) ---
class RouteQuery(BaseModel):
    """Định tuyến câu hỏi của người dùng tới đúng bộ phận xử lý."""
    datasource: str = Field(
        description="Chọn 'vectorstore' nếu câu hỏi liên quan đến quy chế, học vụ, điểm số, đồ án, tốt nghiệp, học bổng của UIT. Chọn 'direct' nếu là câu chào hỏi, giao tiếp thông thường hoặc câu hỏi ngoài luồng."
    )

structured_llm_router = llm.with_structured_output(RouteQuery)

system_router_prompt = """Bạn là chuyên gia phân luồng câu hỏi tối cao của trường Đại học CNTT (UIT).
Nhiệm vụ của bạn là đọc câu hỏi của sinh viên và quyết định chính xác nó thuộc luồng nào:

1. Chọn 'vectorstore': Đối với TẤT CẢ các câu hỏi hỏi về luật lệ, quy chế, điều kiện, học phí, điểm số, chứng chỉ ngoại ngữ (IELTS, TOEIC), đồ án, khóa luận, cảnh cáo học vụ, buộc thôi học, hoặc bất kỳ quy định hành chính nào của UIT.
2. Chọn 'direct': Chỉ dành cho lời chào hỏi (Chào bot, hello), khen ngợi/chửi bới, đặt tên riêng (Fancy), hỏi danh tính của bot, hoặc các câu hỏi hoàn toàn ngoài luồng không liên quan đến đại học (Giá Bitcoin, thời tiết, nấu ăn).

🌟 VÍ DỤ MẪU ĐỂ BẠN HỌC THEO:
- 'Yêu cầu IELTS để tốt nghiệp?' -> Chọn 'vectorstore' (Vì liên quan đến chứng chỉ ngoại ngữ tốt nghiệp)
- 'Bị cảnh cáo học vụ thì sao?' -> Chọn 'vectorstore' (Vì liên quan đến chế tài học vụ)
- 'Bạn tên là gì?' -> Chọn 'direct' (Vì hỏi danh tính)
- 'Thôi học có được xét tốt nghiệp?' -> Chọn 'vectorstore' (Vì liên quan đến quy chế tốt nghiệp)

Câu hỏi cần phân luồng: {question}"""

question_router = ChatPromptTemplate.from_template(system_router_prompt) | structured_llm_router
# Node 1: Tìm kiếm
def retrieve_node(state: GraphState):
    original_question = state["question"]
    search_query = state.get("standalone_question", original_question)
    queries = multi_query_generator(search_query)

    all_docs = []
    seen_content = set()

    # Quét qua tất cả các phiên bản câu hỏi
    for q in queries:
        docs = custom_hybrid_search(q)
        for d in docs:
            if d.page_content not in seen_content:
                all_docs.append(d)
                seen_content.add(d.page_content)

    final_top_docs, confidence = rerank_documents(original_question, all_docs)
    return {
        "documents": final_top_docs,
        "question": original_question,
        "confidence_score": confidence,  # Lưu điểm tin cậy vào state
    }

# Node 2: Đánh giá & Lọc tài liệu
def grade_documents_node(state: GraphState):
    print(f"⏳ [Grader] Đang kiểm định {len(state['documents'])} tài liệu song song (batch)...")
    question = state.get("standalone_question", state["question"])
    documents = state["documents"]

    # --- SỬA LỖI 4: GỌI BATCH THAY VÌ TUẦN TỰ ---
    # Trước: vòng for gọi LLM từng doc một -> N lần gọi API nối tiếp (chậm)
    # Sau:  .batch() gửi toàn bộ N request song song trong 1 lần -> nhanh hơn ~N lần
    inputs = [{"question": question, "context": d.page_content} for d in documents]
    scores = retrieval_grader.batch(inputs)  # Gửi tất cả cùng lúc

    filtered_docs = []
    for d, score in zip(documents, scores):
        if score.binary_score == "yes":
            filtered_docs.append(d)
        else:
            print(f"   ❌ Loại bỏ tài liệu nhiễu (Nguồn: {d.metadata.get('article', 'Không rõ')})")

    print(f"   ✅ Giữ lại {len(filtered_docs)}/{len(documents)} tài liệu hợp lệ.")
    return {"documents": filtered_docs, "question": question}

# ==========================================
# SELF-RAG & CRAG MODULES
# ==========================================

# 1. Self-RAG: Đánh giá độ mơ hồ của câu hỏi
class AmbiguityResult(BaseModel):
    is_ambiguous: bool = Field(description="True nếu câu hỏi cực kỳ mơ hồ (ví dụ: 'điều kiện là gì?', 'bao nhiêu tiền?'), thiếu chủ thể học vụ và chưa được làm rõ.")
    clarification_question: str = Field(description="Nếu mơ hồ, hãy tạo 1 câu hỏi ngắn gọn gợi ý sinh viên cung cấp thêm chi tiết. Nếu rõ ràng, để trống.")

structured_llm_ambiguity = llm.with_structured_output(AmbiguityResult)

system_ambiguity_prompt = """Bạn là chuyên gia phân tích câu hỏi sinh viên UIT.
Đánh giá câu hỏi xem có quá mơ hồ (thiếu hẳn chủ thể quy chế như "điều kiện là gì?", "thủ tục ra sao?") hay không.

LƯU Ý:
1. Nếu câu hỏi là giao tiếp/chào hỏi (VD: "chào bạn", "hello") -> KHÔNG mơ hồ.
2. Nếu câu hỏi có từ khóa học vụ (VD: "học bổng", "điểm F", "học phí", "đồ án", "IELTS") -> KHÔNG mơ hồ.
3. Chỉ coi là mơ hồ khi khuyết hẳn đối tượng tra cứu.

Câu hỏi: {question}"""

ambiguity_checker = ChatPromptTemplate.from_template(system_ambiguity_prompt) | structured_llm_ambiguity

def ambiguity_check_node(state: GraphState):
    question = state.get("standalone_question", state["question"])
    history = state.get("chat_history", [])
    
    if history and len(history) > 0:
        return {"needs_clarification": False}

    if len(question.split()) <= 4 and not any(w in question.lower() for w in ["chào", "hello", "hi", "cảm ơn", "tên"]):
        res = ambiguity_checker.invoke({"question": question})
        if res.is_ambiguous:
            print(f"❓ [Self-RAG] Phát hiện câu hỏi mơ hồ: '{question}'")
            return {
                "needs_clarification": True,
                "clarification_question": res.clarification_question or "Câu hỏi của bạn hiện chưa rõ ràng. Bạn có thể cho tôi biết bạn đang quan tâm đến quy định hay chủ đề học vụ cụ thể nào của UIT không?"
            }

    return {"needs_clarification": False}

def clarify_node(state: GraphState):
    answer = state.get("clarification_question", "Bạn có thể mô tả chi tiết hơn câu hỏi của mình được không?")
    return {"answer": answer, "documents": []}

def check_ambiguity_condition(state: GraphState):
    if state.get("needs_clarification", False):
        return "clarify"
    return "clear"


# 2. CRAG (Corrective RAG): Web Search khi tài liệu nội bộ ít/thiếu
from langchain_core.documents import Document

def web_search_node(state: GraphState):
    print("🌐 [CRAG Web Search] Đang bổ sung tài liệu từ Web (UIT)...")
    question = state.get("standalone_question", state["question"])
    search_query = f"site:uit.edu.vn {question}"
    web_docs = []
    try:
        try:
            from duckduckgo_search import DDGS
            results = list(DDGS().text(search_query, max_results=3))
            if results:
                snippets = "\n".join([r.get("body", "") for r in results if r.get("body")])
                if snippets:
                    web_doc = Document(
                        page_content=f"[Thông tin bổ sung từ Web UIT site:uit.edu.vn]: {snippets}",
                        metadata={"article": "Web UIT (site:uit.edu.vn)", "type": "web"}
                    )
                    web_docs.append(web_doc)
                    print("   ✅ Tìm thấy thông tin web bổ sung thành công.")
        except Exception:
            from langchain_community.tools import DuckDuckGoSearchRun
            search = DuckDuckGoSearchRun()
            res = search.invoke(search_query)
            if res and isinstance(res, str) and len(res.strip()) > 10:
                web_doc = Document(
                    page_content=f"[Thông tin bổ sung từ Cổng thông tin UIT]: {res}",
                    metadata={"article": "Web UIT (site:uit.edu.vn)", "type": "web"}
                )
                web_docs.append(web_doc)
                print("   ✅ Tìm thấy thông tin web bổ sung thành công.")
    except Exception as e:
        print(f"   ⚠️ Lỗi Web Search: {e}")

    existing_docs = state.get("documents", [])
    combined_docs = existing_docs + web_docs
    return {"documents": combined_docs, "is_web_searched": True}


# Điều hướng: Có nên trả lời không (Cập nhật CRAG)?
def check_relevance_condition(state: GraphState):
    print("🔍 [Condition] Kiểm tra kết quả đánh giá tài liệu...")
    documents = state.get("documents", [])
    already_web_searched = state.get("is_web_searched", False)

    if not documents:
        if not already_web_searched:
            print("   ⚠️ Không có tài liệu nội bộ hợp lệ -> Kích hoạt CRAG Web Search")
            return "web_search"
        print("   ❌ Không có tài liệu nào kể cả sau Web Search -> Chuyển sang No-Context Node")
        return "no_context"

    if len(documents) < 2 and not already_web_searched:
        print(f"   ⚠️ Chỉ có {len(documents)} tài liệu nội bộ -> Kích hoạt CRAG Web Search bổ sung")
        return "web_search"

    print("   ✅ Có đủ tài liệu hợp lệ -> Chuyển sang Trạm Tổng Hợp (Generate)")
    return "generate"

def no_context_node(state: GraphState):
    print("   📭 [No-Context] Câu hỏi về quy chế nhưng không tìm được tài liệu phù hợp.")
    question = state["question"]
    return {
        "answer": (
            "Câu hỏi của bạn liên quan đến quy chế học vụ UIT, nhưng tôi không tìm thấy "
            "điều khoản cụ thể nào trong bộ tài liệu hiện có để trả lời chính xác.\n\n"
            "Để đảm bảo thông tin chính xác, bạn vui lòng liên hệ trực tiếp "
            "**Phòng Đào tạo Đại học (P.ĐTĐH)** hoặc tra cứu trên cổng thông tin sinh viên."
        ),
        "documents": [],
        "question": question,
    }

# Node 3: Sinh câu trả lời (Few-Shot)
def generate_node(state: GraphState):
    question = state["question"]
    documents = state["documents"]
    context_text = "\n\n".join([d.page_content for d in documents])

    history = state.get("chat_history", [])
    if history:
        history_str = "\n".join(
            f"{'Sinh viên' if m['role'] == 'user' else 'Bot'}: {m['content']}"
            for m in history
        )
        history_block = f"\n[LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ]:\n{history_str}\n"
    else:
        history_block = ""
    
    system_prompt = """
    [PERSONA]
    Bạn là chuyên viên tư vấn pháp chế và học vụ tài ba của trường Đại học Công nghệ Thông tin (UIT) thuộc ĐHQG-HCM. Bạn có nhiệm vụ giải đáp các thắc mắc về quy chế cho sinh viên một cách chính xác và chuyên nghiệp.

    [NHIỆM VỤ]
    Trả lời thắc mắc của sinh viên dựa trên Ngữ cảnh (Context) được cung cấp. Bạn cần tổng hợp thông tin để đưa ra câu trả lời đầy đủ, rõ ràng nhất.

    [QUY TẮC CỐT LÕI - KHÔNG ĐƯỢC VI PHẠM]
    1. TRUNG THỰC TUYỆT ĐỐI (ZERO-HALLUCINATION): Chỉ đưa vào câu trả lời những thông tin, con số, điều kiện có căn cứ 100% từ ngữ cảnh được cung cấp. Tuyệt đối không tự ý bịa ra dữ kiện, mốc thời gian hoặc hình phạt mới nằm ngoài tài liệu.

    2. LINH HOẠT VỀ NGÔN NGỮ: Bạn ĐƯỢC PHÉP diễn đạt lại (paraphrase), sử dụng từ nối tự nhiên, định dạng danh sách (bullet points) để sinh viên dễ đọc. Bạn NÊN viết tường minh các từ viết tắt học vụ để câu văn trang trọng (Ví dụ: dịch SV -> Sinh viên, ĐTBCTL -> Điểm trung bình chung tích lũy, ĐATN/KLTN -> Đồ án/Khóa luận tốt nghiệp, P.ĐTĐH -> Phòng Đào tạo Đại học).

    3. TRÍCH DẪN RÕ RÀNG: Mọi thông tin, điều khoản quy chế đưa ra phải đi kèm nguồn ở cuối câu hoặc cuối đoạn theo định dạng chính xác: (Nguồn: [Điều khoản] - [Tên văn bản]).

    4. XỬ LÝ XUNG ĐỘT: Nếu các tài liệu trong ngữ cảnh có thông tin mâu thuẫn hoặc khác biệt giữa các hệ đào tạo (Chương trình chuẩn, Chương trình tài năng, Chất lượng cao), hãy liệt kê rõ ràng điều kiện áp dụng của từng văn bản để sinh viên không bị nhầm lẫn.

    5. NHẤT QUÁN VỚI LỊCH SỬ: Nếu có lịch sử hội thoại, hãy đọc kỹ để tránh lặp lại thông tin đã giải thích, và trả lời câu hỏi hiện tại theo đúng mạch cuộc trò chuyện.

    [GIỌNG VĂN & GIAO TIẾP UX]
    - Chuyên nghiệp, rõ ràng, mạch lạc, đi thẳng vào trọng tâm câu hỏi.
    - KIỂM SOÁT LỜI CHÀO: Tuyệt đối KHÔNG tự ý thêm các câu chào hỏi mang tính thủ tục như "Chào bạn!", "Rất vui được hỗ trợ bạn" ở đầu câu trả lời nếu cuộc trò chuyện đang diễn ra liên tục (trừ khi người dùng chủ động chào bạn trước ở câu thoại hiện tại).
    - KHÔNG thêm các câu kết rập khuôn, sáo rỗng ở cuối (Ví dụ: "Nếu có thắc mắc gì hãy liên hệ..."). Hãy dừng lại ngay khi đã trả lời xong xuôi ý chính của quy chế.

    ---------------------
    {history_block}
    [NGỮ CẢNH HIỆN TẠI]: 
    {context}

    [CÂU HỎI CỦA SINH VIÊN]: 
    {question}
    """
    
    prompt = ChatPromptTemplate.from_template(system_prompt)
    generation_chain = prompt | llm.with_config(tags=["user_generation"]) | StrOutputParser()
    
    answer = generation_chain.invoke({
        "context": context_text,
        "question": question,
        "history_block": history_block,
    })
    return {"answer": answer, "documents": documents, "question": question}

# Node 4a: Tái sinh câu trả lời với Prompt nghiêm ngặt hơn (Retry Loop)
def regenerate_node(state: GraphState):
    retry_count = state.get("retry_count", 0) + 1
    print(f"   🔁 [Retry #{retry_count}] Đang tái sinh câu trả lời với ràng buộc Zero-Hallucination nghiêm ngặt hơn...")

    question = state["question"]
    documents = state["documents"]
    context_text = "\n\n".join([d.page_content for d in documents])
    history = state.get("chat_history", [])
    history_block = ""
    if history:
        history_str = "\n".join(
            f"{'Sinh viên' if m['role'] == 'user' else 'Bot'}: {m['content']}"
            for m in history
        )
        history_block = f"\n[LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ]:\n{history_str}\n"

    # Prompt nghiêm khắc hơn, yêu cầu trích dẫn từng câu
    strict_prompt = """
    [CHẾ ĐỘ XÁC MINH NGHIÊM NGẶT - LẦN THỬ #{retry_count}]
    Câu trả lời trước của bạn đã bị phát hiện có thể chứa thông tin không có căn cứ trong tài liệu.
    Hãy viết lại câu trả lời với các ràng buộc SAU, KHÔNG ĐƯỢC BỎ QUA:

    ✅ QUY TẮC BẮT BUỘC (NGHIÊM NGẶT HƠN):
    1. CHỈ sử dụng các con số, điều kiện, mốc thời gian xuất hiện CHÍNH XÁC trong đoạn Context bên dưới.
    2. Mỗi ý trong câu trả lời PHẢI có trích dẫn nguồn (Nguồn: [Điều ...] - [Tên văn bản]).
    3. Nếu không tìm thấy thông tin cụ thể trong Context, hãy nói rõ "Quy chế không quy định cụ thể" thay vì tự suy diễn.
    4. KHÔNG thêm bất kỳ thông tin nào ngoài Context, kể cả từ kiến thức nền.

    ---------------------
    {history_block}
    [NGỮ CẢNH HIỆN TẠI]:
    {context}

    [CÂU HỎI CỦA SINH VIÊN]:
    {question}
    """

    prompt = ChatPromptTemplate.from_template(strict_prompt)
    chain = prompt | llm.with_config(tags=["user_generation"]) | StrOutputParser()
    answer = chain.invoke({
        "context": context_text,
        "question": question,
        "history_block": history_block,
        "retry_count": retry_count,
    })
    return {"answer": answer, "documents": documents, "question": question, "retry_count": retry_count}

# Node 4b: Trạm An Toàn cuối cùng (Fallback sau khi đã retry hết lần)
def fallback_node(state: GraphState):
    retry_count = state.get("retry_count", 0)
    print(f"   🛡️ [Safety] Đã hết {retry_count} lần thử - Chặn câu trả lời bịa đặt!")
    return {"answer": "Dựa trên quy chế hiện tại, tôi đã tìm thấy một vài tài liệu liên quan nhưng không thể đưa ra kết luận chắc chắn sau nhiều lần kiểm định. Để đảm bảo chính xác, bạn vui lòng liên hệ trực tiếp **Phòng Đào Tạo Đại học (P.ĐTĐH)** để được giải đáp chính thức."}

# Node 5: Trạm Trả lời Trực tiếp (Không cần tìm kiếm)
def direct_answer_node(state: GraphState):
    print("--- TRẠM XỬ LÝ NHANH (KHÔNG DÙNG RAG) ---")
    question = state["question"]
    history = state.get("chat_history", [])
    history_str = ""
    for msg in history:
        history_str += f"{msg['role']}: {msg['content']}\n"

    prompt = f"""Bạn là trợ lý học vụ UIT. Dựa vào lịch sử chat bên dưới, hãy trả lời câu hỏi tiếp theo một cách tự nhiên, mạch lạc, tránh lặp lại lời chào hỏi không cần thiết.
    
    Lịch sử cuộc trò chuyện:
    {history_str}
    
    Câu hỏi mới nhất: {question}
    Trả lời:"""
    
    history_block = ""
    if history:
        history_str = "\n".join(
            f"{'Sinh viên' if m['role'] == 'user' else 'Bot'}: {m['content']}"
            for m in history
        )
        history_block = f"\n[LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ]:\n{history_str}\n"

    # Prompt nghiêm khắc hơn, yêu cầu trích dẫn từng câu
    strict_prompt = """
    [CHẾ ĐỘ XÁC MINH NGHIÊM NGẶT - LẦN THỬ #{retry_count}]
    Câu trả lời trước của bạn đã bị phát hiện có thể chứa thông tin không có căn cứ trong tài liệu.
    Hãy viết lại câu trả lời với các ràng buộc SAU, KHÔNG ĐƯỢC BỎ QUA:

    ✅ QUY TẮC BẮT BUỘC (NGHIÊM NGẶT HƠN):
    1. CHỈ sử dụng các con số, điều kiện, mốc thời gian xuất hiện CHÍNH XÁC trong đoạn Context bên dưới.
    2. Mỗi ý trong câu trả lời PHẢI có trích dẫn nguồn (Nguồn: [Điều ...] - [Tên văn bản]).
    3. Nếu không tìm thấy thông tin cụ thể trong Context, hãy nói rõ "Quy chế không quy định cụ thể" thay vì tự suy diễn.
    4. KHÔNG thêm bất kỳ thông tin nào ngoài Context, kể cả từ kiến thức nền.

    ---------------------
    {history_block}
    [NGỮ CẢNH HIỆN TẠI]:
    {context}

    [CÂU HỎI CỦA SINH VIÊN]:
    {question}
    """

    prompt = ChatPromptTemplate.from_template(strict_prompt)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context_text,
        "question": question,
        "history_block": history_block,
        "retry_count": retry_count,
    })
    return {"answer": answer, "documents": documents, "question": question, "retry_count": retry_count}

# Node 4b: Trạm An Toàn cuối cùng (Fallback sau khi đã retry hết lần)
def fallback_node(state: GraphState):
    retry_count = state.get("retry_count", 0)
    print(f"   🛡️ [Safety] Đã hết {retry_count} lần thử - Chặn câu trả lời bịa đặt!")
    return {"answer": "Dựa trên quy chế hiện tại, tôi đã tìm thấy một vài tài liệu liên quan nhưng không thể đưa ra kết luận chắc chắn sau nhiều lần kiểm định. Để đảm bảo chính xác, bạn vui lòng liên hệ trực tiếp **Phòng Đào Tạo Đại học (P.ĐTĐH)** để được giải đáp chính thức."}
# Node 5: Trạm Trả lời Trực tiếp (Không cần tìm kiếm)
def direct_answer_node(state: GraphState):
    print("--- TRẠM XỬ LÝ NHANH (KHÔNG DÙNG RAG) ---")
    question = state["question"]
    history = state.get("chat_history", [])
    history_str = ""
    for msg in history:
        history_str += f"{msg['role']}: {msg['content']}\n"

    prompt = f"""Bạn là trợ lý học vụ UIT. Dựa vào lịch sử chat bên dưới, hãy trả lời câu hỏi tiếp theo một cách tự nhiên, mạch lạc, tránh lặp lại lời chào hỏi không cần thiết.
    
    Lịch sử cuộc trò chuyện:
    {history_str}
    
    Câu hỏi mới nhất: {question}
    Trả lời:"""
    
    response = llm.invoke(prompt)
    return {"answer": response.content, "documents": [], "question": question}

# Hàm logic của Cảnh sát giao thông (Đặt ở cửa START)
def route_question(state: GraphState):
    print("🚦 [Router] Đang phân tích ý định của người dùng...")
    question_to_route = state.get("standalone_question", state["question"])
    source = question_router.invoke({"question": question_to_route})
    
    if source.datasource == "vectorstore":
        print("   👉 Luồng RAG: Đang chuyển hướng đi tìm tài liệu...")
        return "vectorstore"
    elif source.datasource == "direct":
        print("   👉 Luồng Direct: Chuyển hướng sang giao tiếp trực tiếp...")
        return "direct"

# Điều hướng: Kiểm tra Ảo giác sau khi Generate / Regenerate
MAX_RETRIES = 2

def check_hallucination_condition(state: GraphState):
    retry_count = state.get("retry_count", 0)
    print(f"🔎 [Verifier] Kiểm tra ảo giác (lần thử hiện tại: {retry_count + 1}/{MAX_RETRIES + 1})...")
    question = state["question"]
    documents = state["documents"]
    generation = state["answer"]

    context_text = "\n\n".join([d.page_content for d in documents])

    score = hallucination_grader.invoke({"question": question, "documents": context_text, "generation": generation})
    print(f"   🧠 [Lý do của Giám thị]: {score.reasoning}")

    if score.binary_score == "yes":
        print("   ✅ Câu trả lời hợp lệ, không có ảo giác.")
        return "useful"
    else:
        if retry_count < MAX_RETRIES:
            print(f"   ⚠️ PHÁT HIỆN ẢO GIÁC! Thử tái sinh lần {retry_count + 1}/{MAX_RETRIES}...")
            return "retry"
        else:
            print(f"   🚨 Đã vượt quá {MAX_RETRIES} lần thử - Kích hoạt Safety Fallback.")
            return "hallucinated"
    
# ==========================================
# PHẦN 4: LẮP RÁP ĐỒ THỊ VÀ CHẠY
# ==========================================
workflow = StateGraph(GraphState)

# 1. Đăng ký TẤT CẢ các Trạm
workflow.add_node("reformulate", reformulate_node)
workflow.add_node("ambiguity_check", ambiguity_check_node) # MỚI: Self-RAG
workflow.add_node("clarify", clarify_node)                 # MỚI: Self-RAG
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_documents_node)
workflow.add_node("web_search", web_search_node)           # MỚI: CRAG
workflow.add_node("generate", generate_node)
workflow.add_node("regenerate", regenerate_node)
workflow.add_node("fallback", fallback_node)
workflow.add_node("direct_answer", direct_answer_node)
workflow.add_node("no_context", no_context_node)

# 2. Xây dựng đường đi
workflow.add_edge(START, "reformulate")
workflow.add_edge("reformulate", "ambiguity_check")

# Self-RAG & Router: Đánh giá câu hỏi mơ hồ & Định tuyến
def check_ambiguity_and_route(state: GraphState):
    if state.get("needs_clarification", False):
        return "clarify"
    return route_question(state)

workflow.add_conditional_edges(
    "ambiguity_check",
    check_ambiguity_and_route,
    {
        "clarify": "clarify",
        "vectorstore": "retrieve",
        "direct": "direct_answer",
    }
)
workflow.add_edge("clarify", END)

# Nhánh RAG (Vectorstore) + CRAG Web Search
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges(
    "grade",
    check_relevance_condition,
    {
        "generate": "generate",
        "web_search": "web_search",  # MỚI: CRAG Web search
        "no_context": "no_context",
    }
)

workflow.add_edge("web_search", "generate") # MỚI: Web search xong qua Generate

# Điểm kiểm tra ảo giác
workflow.add_conditional_edges(
    "generate",
    check_hallucination_condition,
    {
        "useful": END,
        "retry": "regenerate",
        "hallucinated": "fallback",
    }
)

workflow.add_conditional_edges(
    "regenerate",
    check_hallucination_condition,
    {
        "useful": END,
        "retry": "regenerate",
        "hallucinated": "fallback",
    }
)

workflow.add_edge("fallback", END)
workflow.add_edge("no_context", END)
workflow.add_edge("direct_answer", END)

# 3. Đóng gói hệ thống với Session Memory
memory_checkpointer = MemorySaver()
app = workflow.compile(checkpointer=memory_checkpointer)

if __name__ == "__main__":
    print("\n🚀 Chatbot UIT (Agentic RAG) đã khởi động!")
    while True:
        user_input = input("\nSinh viên: ")
        if user_input.lower() in ['exit', 'quit']: break
        
        inputs = {"question": user_input}
        final_state = app.invoke(inputs)
        
        if "answer" in final_state:
            print(f"Bot: {final_state['answer']}")
        else:
            print("Bot: Dựa trên bộ quy chế hiện tại, tôi chưa tìm thấy thông tin để trả lời câu hỏi này.")