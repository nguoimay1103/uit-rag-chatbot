import os
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle
from langchain_community.retrievers import BM25Retriever
from langchain_experimental.text_splitter import SemanticChunker

load_dotenv()

def ingest_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../data/raw"))
    PROCESSED_DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../data/processed"))
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    csv_files = [os.path.join(RAW_DATA_DIR, f) for f in ["train.csv", "val.csv", "test.csv"]]
    all_data = [] # Gom tất cả dataframe vào đây
    
    for file in csv_files:
        if os.path.exists(file):
            df = pd.read_csv(file)
            # Chỉ lấy các cột cần thiết làm tri thức
            df = df[['context', 'article', 'document']] 
            df = df.dropna(subset=['context'])
            all_data.append(df)

    if not all_data:
        print("⚠️ Không tìm thấy file dữ liệu CSV nào trong data/raw!")
        return

    # 1. GỘP CẢ 3 FILE LẠI THÀNH 1 BẢNG LỚN
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 2. XÓA CÁC ĐOẠN CONTEXT TRÙNG LẶP (Bước cực kỳ quan trọng)
    combined_df = combined_df.drop_duplicates(subset=['context'])
    
    print(f"Tổng số đoạn quy chế duy nhất sau khi lọc trùng lặp: {len(combined_df)}")

    # 3. Chuyển thành LangChain Document
    documents = []
    for _, row in combined_df.iterrows():
        metadata = {
            "document": str(row.get('document', '')),
            "article": str(row.get('article', ''))
        }
        documents.append(Document(page_content=str(row['context']), metadata=metadata))

    # Khởi tạo OpenAI Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Dùng Semantic Chunker thay cho Recursive
    print("Đang cắt dữ liệu theo ngữ nghĩa (Semantic Chunking)...")
    semantic_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile", # Cắt khi độ lệch ngữ nghĩa vượt quá ngưỡng phần trăm nhất định
        breakpoint_threshold_amount=80
    )
    splits = semantic_splitter.split_documents(documents)
    print(f"Đã cắt thành {len(splits)} chunks ngữ nghĩa hoàn chỉnh.")

    print("1. Đang đẩy dữ liệu lên Qdrant Cloud...")
    QdrantVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name="uit_admissions",
        force_recreate=True,
    )
    
    print("2. Đang tạo và lưu bộ tìm kiếm từ khóa BM25...")
    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 3 
    
    # Lưu file BM25 xuống thư mục data/processed
    bm25_output_path = os.path.join(PROCESSED_DATA_DIR, "bm25_retriever.pkl")
    with open(bm25_output_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
    print(f"Đã lưu BM25 retriever tại: {bm25_output_path}")

    print("--- Hoàn tất toàn bộ quá trình nạp dữ liệu! ---")

if __name__ == "__main__":
    ingest_data()