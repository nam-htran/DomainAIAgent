# app.py - Streamlit UI for Domain-AI Copilot (Final Version)

import streamlit as st
from rag_engine.embedding import get_embedding_cached
from rag_engine.vector_store import query_vector_store, init_qdrant
from rag_engine.llm import call_llm_cached, suggest_followups, create_standalone_query_from_history
from rag_engine.reranker import rerank_with_cohere
from rag_engine.data_loader import process_and_load_files

# ----- Cấu hình trang và tiêu đề -----
st.set_page_config(
    page_title="AI Copilot - RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)
st.title("🤖 AI Copilot – Chatbot Hỏi-Đáp Thông Minh")

# ----- Khởi tạo cơ sở dữ liệu vector -----
init_qdrant()

# ----- Sidebar để tải và xử lý tài liệu -----
with st.sidebar:
    st.header("📚 Quản lý Cơ sở Tri thức")
    uploaded_files = st.file_uploader(
        "Tải lên các file PDF, DOCX, hoặc TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    if st.button("Xử lý và Nạp vào CSDL", disabled=not uploaded_files, use_container_width=True):
        with st.spinner("Đang xử lý..."):
            stats = process_and_load_files(uploaded_files)
            st.success(f"Hoàn tất! Đã thêm {stats['new_chunks_added']} đoạn nội dung mới.")

# ----- Quản lý và Hiển thị Lịch sử Chat -----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# ----- Xử lý khi người dùng nhập câu hỏi mới -----
if query := st.chat_input("Hãy hỏi tôi bất cứ điều gì..."):
    st.chat_message("user").markdown(query)
    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.spinner("🧠 AI đang suy nghĩ..."):
        # ======================================================================
        # BƯỚC 0: VIẾT LẠI CÂU HỎI DỰA TRÊN LỊCH SỬ (HISTORY-AWARE)
        # ======================================================================
        if len(st.session_state.chat_history) > 1:
            standalone_query = create_standalone_query_from_history(st.session_state.chat_history, query)
            # Hiển thị câu hỏi đã được diễn giải để người dùng biết
            if standalone_query.lower() != query.lower():
                st.info(f"Đã diễn giải câu hỏi thành: *\"{standalone_query}\"*")
        else:
            standalone_query = query

        # BƯỚC 1: EMBEDDING CÂU HỎI ĐỘC LẬP
        query_embedding = query_embedding = get_embedding_cached(standalone_query)

        # BƯỚC 2 & 3: TRUY XUẤT VÀ SẮP XẾP LẠI
        retrieved_docs = query_vector_store(query_embedding, top_k=10)
        reranked_docs = rerank_with_cohere(query=standalone_query, documents=retrieved_docs, top_n=5) if retrieved_docs else []

        # ======================================================================
        # BƯỚC 4 & 5: TẠO PROMPT VÀ SINH CÂU TRẢ LỜI (CÓ LOGIC FALLBACK)
        # ======================================================================
        if not reranked_docs:
            # ------- KỊCH BẢN 1: KHÔNG TÌM THẤY TÀI LIỆU (FALLBACK TO LLM) -------
            st.warning("Không tìm thấy thông tin trong tài liệu. Trả lời dựa trên kiến thức chung.")
            system_prompt = "Bạn là một trợ lý AI hữu ích. Hãy trả lời câu hỏi của người dùng một cách trực tiếp."
            final_prompt = standalone_query
            answer = call_llm_cached(final_prompt, system_prompt)
            
            st.chat_message("assistant").markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
        else:
            # ------- KỊCH BẢN 2: TÌM THẤY TÀI LIỆU (RAG) -------
            context = "\n\n---\n\n".join([doc.payload.get("text", "") for doc in reranked_docs])
            system_prompt = "Dựa vào ngữ cảnh được cung cấp dưới đây để trả lời câu hỏi của người dùng một cách chính xác và chi tiết. Nếu thông tin không có trong ngữ cảnh, hãy nói rằng bạn không tìm thấy thông tin trong tài liệu đã cung cấp."
            final_prompt = f"**Ngữ cảnh:**\n{context}\n\n**Câu hỏi:**\n{standalone_query}"
            
            answer = call_llm_cached(final_prompt, system_prompt)
            st.chat_message("assistant").markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

            # Hiển thị nguồn trích dẫn
            with st.expander("📚 Xem nguồn trích dẫn"):
                for i, doc in enumerate(reranked_docs):
                    metadata = doc.payload or {}
                    source = metadata.get("source", "Không rõ nguồn")
                    st.markdown(f"**[{i+1}] Nguồn:** `{source}`")
                    st.caption(f"```{metadata.get('text', '')[:200]}...```")

        # Gợi ý câu hỏi tiếp theo
        with st.expander("💡 Gợi ý câu hỏi tiếp theo"):
            suggestions = suggest_followups(answer)
            for sug in suggestions:
                st.markdown(f"- {sug}")