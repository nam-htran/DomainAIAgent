# rag_engine/embedding.py
import os
from openai import OpenAI # THAY ĐỔI: Quay lại sử dụng client OpenAI
from dotenv import load_dotenv
import hashlib
import pickle
from typing import List

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
# THAY ĐỔI: Sử dụng model mới nhất. 'light-auto' là lựa chọn tốt, cân bằng giữa hiệu năng và chi phí.
MODEL_EMBEDDING = os.getenv("MODEL_EMBEDDING", "embed-v4.0-light-auto")

CACHE_DIR = ".cache/embeddings"
os.makedirs(CACHE_DIR, exist_ok=True)

# THAY ĐỔI: Khởi tạo client OpenAI và trỏ đến lớp tương thích của Cohere
client = OpenAI(
    base_url="https://api.cohere.ai/compatibility/v1",
    api_key=COHERE_API_KEY,
)

def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _cache_path(text: str) -> str:
    return os.path.join(CACHE_DIR, _hash_text(text) + ".pkl")

def get_embedding(texts: List[str], model=MODEL_EMBEDDING) -> List[list[float]]:
    """
    Lấy embedding cho một danh sách văn bản, sử dụng client tương thích OpenAI.
    Model embed-v4.0-auto tự động xử lý input_type.
    """
    # THAY ĐỔI: Sử dụng client.embeddings.create theo chuẩn OpenAI
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    # Trích xuất embedding từ cấu trúc response mới
    return [d.embedding for d in response.data]

def get_embedding_cached(text: str, model=MODEL_EMBEDDING) -> list[float]:
    """Lấy embedding cho MỘT văn bản từ cache hoặc API."""
    path = _cache_path(text)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
            
    # THAY ĐỔI: Gọi hàm get_embedding mới (không còn input_type)
    embedding = get_embedding([text], model)[0]
    
    with open(path, "wb") as f:
        pickle.dump(embedding, f)
        
    return embedding