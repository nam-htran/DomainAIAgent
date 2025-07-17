# rag_engine/reranker.py
import os
import cohere
from dotenv import load_dotenv

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

def rerank_with_cohere(query: str, documents: list, top_n: int = 5):
    docs_text = [doc.payload.get("text", "") for doc in documents]
    if not docs_text:
        return []
    results = co.rerank(
        model="rerank-multilingual-v3.0",
        query=query,
        documents=docs_text,
        top_n=top_n,
    )
    return [documents[r.index] for r in results.results]