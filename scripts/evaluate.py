import os
import sys
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "../backend"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if os.path.join(BACKEND_DIR, "app") not in sys.path:
    sys.path.insert(0, os.path.join(BACKEND_DIR, "app"))

try:
    from app.chatbot import app as rag_agent
except ImportError:
    from chatbot import app as rag_agent

def load_test_data(json_path):
    print(f"📂 Đang đọc dữ liệu từ file: {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"✅ Đã tải thành công {len(test_data)} câu hỏi!")
    return test_data

def run_evaluation(test_data):
    print(f"🚀 Bắt đầu cho hệ thống RAG thi thử...")
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for idx, item in enumerate(test_data):
        q = item["question"]
        gt = item["ground_truth"]
        
        print(f"[{idx+1}/{len(test_data)}] Đang hỏi: {q}")
        
        try:
            # Gọi hệ thống RAG hiện tại của bạn
            result = rag_agent.invoke({
                "question": q,
                "chat_history": [] 
            })
            
            # Trích xuất dữ liệu
            generated_answer = result.get("answer", "")
            retrieved_docs = [doc.page_content for doc in result.get("documents", [])]
            
            questions.append(q)
            answers.append(generated_answer)
            contexts.append(retrieved_docs)
            ground_truths.append(gt)
            
        except Exception as e:
            print(f"⚠️ Lỗi ở câu '{q}': {e}")
            continue

    # Đóng gói thành định dạng HuggingFace Dataset cho Ragas
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data_dict)

    print("⚖️ Đang giao cho Giám khảo Ragas chấm điểm... (sẽ mất vài phút)")
    
    # Thực hiện đánh giá
    result = evaluate(
        dataset = dataset, 
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
    )

    # Xuất báo cáo
    REPORT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../reports"))
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, "rag_evaluation_report.csv")
    df_result = result.to_pandas()
    df_result.to_csv(report_path, index=False, encoding="utf-8-sig")
    
    print("\n" + "="*50)
    print("✅ ĐÃ CHẤM XONG! KẾT QUẢ TỔNG QUAN:")
    print(result)
    print("="*50)
    print(f"📁 Đã xuất bảng điểm chi tiết tại: {report_path}")

if __name__ == "__main__":
    test_json_path = os.path.abspath(os.path.join(BASE_DIR, "../data/test/test_dataset.json"))
    my_test_data = load_test_data(test_json_path)
    run_evaluation(my_test_data)