# rag_engine/llm.py
import os
from openai import OpenAI
from dotenv import load_dotenv
import hashlib
import pickle

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
CACHE_DIR = ".cache/llm"
os.makedirs(CACHE_DIR, exist_ok=True)
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-chat-v3-0324:free")

def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

def _cache_path(prompt: str) -> str:
    return os.path.join(CACHE_DIR, _hash_prompt(prompt) + ".pkl")

def call_llm(prompt: str, system_prompt: str = "Bạn là một trợ lý AI hữu ích.", model=LLM_MODEL) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content

def call_llm_cached(prompt: str, system_prompt: str = "Bạn là một trợ lý AI hữu ích.", model=LLM_MODEL) -> str:
    # Hash bao gồm cả system_prompt để cache chính xác hơn
    cache_key = prompt + system_prompt + model
    path = _cache_path(cache_key)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    result = call_llm(prompt, system_prompt, model)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return result

def suggest_followups(answer: str) -> list[str]:
    prompt = f"Dựa vào câu trả lời sau, hãy gợi ý 3 câu hỏi tiếp theo ngắn gọn mà người dùng có thể muốn hỏi:\n\n{answer}"
    result = call_llm(prompt, model="mistralai/mistral-7b-instruct:free")
    suggestions = [line.strip("-•* ").strip() for line in result.strip().split("\n") if line.strip()]
    return suggestions[:3]

# ==============================================================================
# MỚI: HÀM VIẾT LẠI CÂU HỎI DỰA TRÊN LỊCH SỬ
# ==============================================================================
def create_standalone_query_from_history(chat_history: list, new_query: str) -> str:
    """
    Sử dụng LLM để biến một câu hỏi phụ thuộc vào ngữ cảnh thành một câu hỏi độc lập.
    """
    # Chỉ lấy 3 lượt chat gần nhất để prompt không quá dài
    history_context = "\n".join(
        [f"{turn['role']}: {turn['content']}" for turn in chat_history[-3:]]
    )

    prompt = f"""
    Dựa vào lịch sử hội thoại dưới đây và câu hỏi theo sau, hãy viết lại câu hỏi theo sau thành một câu hỏi độc lập, đầy đủ ý nghĩa mà không cần đến lịch sử hội thoại.

    **Lịch sử hội thoại:**
    {history_context}

    **Câu hỏi theo sau:** "{new_query}"

    **Yêu cầu:**
    - Nếu câu hỏi theo sau đã đủ ý nghĩa, chỉ cần trả về chính nó.
    - Nếu câu hỏi theo sau là một câu hỏi ngắn hoặc tham chiếu (ví dụ: "Kể thêm đi", "Tại sao vậy?", "Nó là gì?"), hãy viết lại nó một cách cụ thể.
    - Chỉ trả về duy nhất câu hỏi đã được viết lại, không thêm bất kỳ lời giải thích nào.

    **Câu hỏi độc lập:**
    """
    
    # Sử dụng một model nhanh cho tác vụ này
    standalone_query = call_llm(
        prompt,
        system_prompt="Bạn là một chuyên gia viết lại câu hỏi AI.",
        model="mistralai/mistral-7b-instruct:free"
    )
    return standalone_query.strip()