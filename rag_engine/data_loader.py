# rag_engine/data_loader.py
import hashlib
import uuid  # THÊM: Import thư viện uuid
from typing import List, Dict, Any
from .file_processor import parse_file, smart_chunk
from .embedding import get_embedding
from .vector_store import client as qdrant_client, COLLECTION_NAME
from qdrant_client import models

def process_and_load_files(files: List[Any]) -> Dict[str, int]:
    """
    Xử lý một danh sách file tải lên, chunk, kiểm tra sự tồn tại,
    embedding và nạp vào Qdrant.
    """
    all_chunks_with_metadata = []
    for file in files:
        text = parse_file(file)
        chunks = smart_chunk(text)
        for chunk_text in chunks:
            all_chunks_with_metadata.append({"text": chunk_text, "source": file.name})

    if not all_chunks_with_metadata:
        return {"total_files": len(files), "new_chunks_added": 0, "skipped_chunks": 0}

    # ==============================================================================
    # THAY ĐỔI LOGIC TẠO ID
    # ==============================================================================
    # Bước 3: Tạo ID duy nhất và tất định dưới dạng UUID
    
    # Tạo một namespace cố định cho project của chúng ta
    namespace_uuid = uuid.NAMESPACE_DNS
    
    chunk_ids = [
        str(uuid.uuid5(namespace_uuid, chunk['text'])) 
        for chunk in all_chunks_with_metadata
    ]

    # Kiểm tra sự tồn tại bằng các UUID này
    try:
        existing_points = qdrant_client.retrieve(
            collection_name=COLLECTION_NAME, ids=chunk_ids, with_payload=False, with_vectors=False
        )
        existing_ids = {point.id for point in existing_points}
    except Exception:
        existing_ids = set()

    # Bước 4: Lọc ra những chunk thực sự mới
    truly_new_chunks = []
    for i, chunk_meta in enumerate(all_chunks_with_metadata):
        if chunk_ids[i] not in existing_ids:
            chunk_meta['id'] = chunk_ids[i] # Gán UUID làm ID
            truly_new_chunks.append(chunk_meta)

    skipped_count = len(all_chunks_with_metadata) - len(truly_new_chunks)
    if not truly_new_chunks:
        return {"total_files": len(files), "new_chunks_added": 0, "skipped_chunks": skipped_count}
        
    # Bước 5: Embedding các chunk mới
    new_texts = [chunk['text'] for chunk in truly_new_chunks]
    new_embeddings = get_embedding(new_texts)

    # Bước 6: Nạp vào Qdrant với ID là UUID
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=models.Batch(
            ids=[chunk['id'] for chunk in truly_new_chunks], # Sử dụng UUIDs
            vectors=new_embeddings,
            payloads=[{"text": chunk['text'], "source": chunk['source']} for chunk in truly_new_chunks]
        ),
        wait=True
    )
    
    return {
        "total_files": len(files),
        "new_chunks_added": len(truly_new_chunks),
        "skipped_chunks": skipped_count
    }