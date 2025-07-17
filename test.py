# test_rag.py

from rag_engine.embedding import get_embedding
from rag_engine.vector_store import query_vector_store
from rag_engine.llm import call_llm, count_tokens, suggest_followups
from rag_engine.reranker import rerank_results
import os


def main():
    print("=== Domain-AI Copilot (Phase 2) ===")
    while True:
        query = input("\n🔍 Nhập câu hỏi của bạn: ")
        if query.lower() in ["exit", "quit"]:
            break

        query_embedding = get_embedding(query)
        top_k_results = query_vector_store(query_embedding, top_k=10)

        # Re-rank các kết quả
        reranked = rerank_results(query, top_k_results, top_n=5)

        print("\n📄 Top 5 tài liệu được truy xuất (sau khi rerank):")
        context = ""
        for i, doc in enumerate(reranked):
            metadata = doc.payload or {}
            content = metadata.get("text", "<no content>")
            score = doc.score
            source = metadata.get("source", "unknown")
            chunk_id = metadata.get("chunk_id", "?")
            print(f"[{i+1}] (Score: {score:.4f}) - {source} [Chunk #{chunk_id}]\n{content}\n")
            context += content + "\n"

        token_count = count_tokens(context)
        print(f"🧠 Tổng số tokens dùng làm context: {token_count}")

        final_prompt = f"Trả lời câu hỏi sau dựa vào thông tin sau:\n---\n{context}\n---\nCâu hỏi: {query}"
        answer = call_llm(final_prompt)

        print("\n🤖 Trả lời:")
        print(answer)

        suggestions = suggest_followups(answer)
        print("\n💡 Gợi ý câu hỏi tiếp theo:")
        for i, sug in enumerate(suggestions, 1):
            print(f"{i}. {sug}")


if __name__ == "__main__":
    main()
