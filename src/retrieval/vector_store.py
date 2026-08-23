import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
from src.config import CHROMA_DB_DIR, EMBEDDING_MODEL_NAME

class ManualVectorStore:
    def __init__(self, collection_name: str = "manual_chunks"):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return

        ids = [chunk.get("id") or chunk.get("chunk_id") for chunk in chunks]
        documents = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"[VectorStore] Successfully indexed {len(chunks)} chunks into ChromaDB.")

    def search(self, query: str, user_id: str = "demo_user", top_k: int = 8) -> Dict[str, Any]:
        return self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"user_id": user_id}
        )