"""
vector_store.py
----------------
Retrieval layer: hybrid dense (embedding) + sparse (BM25 keyword) search,
fused with Reciprocal Rank Fusion, optionally hard-filtered by a
deterministic numeric-range match, then reranked with a cross-encoder.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from src.config import (
    BM25_CANDIDATE_POOL,
    CHROMA_DB_DIR,
    DENSE_CANDIDATE_POOL,
    EMBEDDING_MODEL_NAME,
    RERANK_CANDIDATE_POOL,
    RERANKER_MODEL_NAME,
    RRF_K,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _parse_scoped_ranges(raw: str) -> List[Tuple[int, int]]:
    if not raw:
        return []
    ranges = []
    for part in raw.split("|"):
        if "-" not in part:
            continue
        s, e = part.split("-", 1)
        try:
            ranges.append((int(s), int(e)))
        except ValueError:
            continue
    return ranges


class ManualVectorStore:
    def __init__(self, collection_name: str = "manual_chunks"):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            # Uses RERANKER_MODEL_NAME if defined in config, or falls back to ms-marco-MiniLM-L-6-v2
            model_to_load = globals().get("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            if not model_to_load or model_to_load == "BAAI/bge-reranker-base":
                model_to_load = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            self._reranker = CrossEncoder(model_to_load, max_length=512)
        return self._reranker

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return
        ids = [chunk.get("id") or chunk.get("chunk_id") for chunk in chunks]
        documents = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        print(f"[VectorStore] Successfully indexed {len(chunks)} chunks into ChromaDB.")

    def _scoped_docs(self, user_id: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
        where = {"user_id": user_id} if not doc_id else {
            "$and": [{"user_id": user_id}, {"doc_id": doc_id}]
        }
        return self.collection.get(where=where, include=["documents", "metadatas"])

    def _dense_search(self, query: str, user_id: str, doc_id: Optional[str], n: int) -> List[Tuple[str, str, dict]]:
        where = {"user_id": user_id} if not doc_id else {
            "$and": [{"user_id": user_id}, {"doc_id": doc_id}]
        }
        res = self.collection.query(query_texts=[query], n_results=n, where=where)
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        return list(zip(ids, docs, metas))

    def _bm25_search(self, query: str, user_id: str, doc_id: Optional[str], n: int) -> List[Tuple[str, str, dict]]:
        pool = self._scoped_docs(user_id, doc_id)
        ids = pool.get("ids", [])
        docs = pool.get("documents", [])
        metas = pool.get("metadatas", [])
        if not docs:
            return []

        tokenized_corpus = [_tokenize(d) for d in docs]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(_tokenize(query))

        ranked = sorted(zip(ids, docs, metas, scores), key=lambda x: x[3], reverse=True)
        return [(i, d, m) for i, d, m, s in ranked[:n]]

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[str, str, dict]],
        bm25_results: List[Tuple[str, str, dict]],
        k: int = RRF_K,
    ) -> List[Tuple[str, str, dict, float]]:
        scores: Dict[str, float] = {}
        payload: Dict[str, Tuple[str, str, dict]] = {}

        for rank, (cid, doc, meta) in enumerate(dense_results):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            payload[cid] = (cid, doc, meta)

        for rank, (cid, doc, meta) in enumerate(bm25_results):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            payload[cid] = (cid, doc, meta)

        fused = [(payload[cid][0], payload[cid][1], payload[cid][2], score) for cid, score in scores.items()]
        fused.sort(key=lambda x: x[3], reverse=True)
        return fused

    def _apply_range_filter(
        self, candidates: List[Tuple[str, str, dict, float]], query: str
    ) -> List[Tuple[str, str, dict, float]]:
        query_numbers = [int(n) for n in re.findall(r"\b\d{2,6}\b", query)]
        if not query_numbers:
            return candidates

        kept = []
        excluded_any = False
        for cid, doc, meta, score in candidates:
            ranges = _parse_scoped_ranges(meta.get("scoped_ranges", ""))
            if not ranges:
                kept.append((cid, doc, meta, score))
                continue
            matches = any(start <= qn <= end for qn in query_numbers for start, end in ranges)
            if matches:
                kept.append((cid, doc, meta, score))
            else:
                excluded_any = True

        if excluded_any and kept:
            return kept
        return candidates

    def _rerank(self, query: str, candidates: List[Tuple[str, str, dict, float]], top_k: int) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        pool = candidates[:RERANK_CANDIDATE_POOL]
        pairs = [[query, doc] for _, doc, _, _ in pool]
        try:
            cross_scores = self.reranker.predict(pairs)
        except Exception as e:
            print(f"[Reranker Warning] falling back to fusion order: {e}")
            cross_scores = [score for _, _, _, score in pool]

        reranked = sorted(zip(pool, cross_scores), key=lambda x: x[1], reverse=True)
        results = []
        for (cid, doc, meta, _fusion_score), rerank_score in reranked[:top_k]:
            results.append({
                "id": cid,
                "content": doc,
                "metadata": meta,
                "relevance_score": float(rerank_score),
            })
        return results

    def search(self, query: str, user_id: str = "demo_user", doc_id: Optional[str] = None, top_k: int = 8) -> List[Dict[str, Any]]:
        dense = self._dense_search(query, user_id, doc_id, n=DENSE_CANDIDATE_POOL)
        bm25 = self._bm25_search(query, user_id, doc_id, n=BM25_CANDIDATE_POOL)

        if not dense and not bm25:
            return []

        fused = self._reciprocal_rank_fusion(dense, bm25)
        filtered = self._apply_range_filter(fused, query)
        return self._rerank(query, filtered, top_k)