# rag_engine/vector_store.py
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
import os

load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my-rag-collection")
EMBEDDING_DIM = os.getenv("EMBEDDING_DIM")

client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)

def init_qdrant(vector_size=EMBEDDING_DIM):
    try:
        client.get_collection(collection_name=COLLECTION_NAME)
    except Exception:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

def query_vector_store(embedding: list[float], top_k: int = 5):
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        limit=top_k,
        with_payload=True,
    )
    return hits