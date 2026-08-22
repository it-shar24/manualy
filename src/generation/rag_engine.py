import requests
import json
from typing import Dict, Any, List
from src.retrieval.vector_store import ManualVectorStore
from src.config import OLLAMA_BASE_URL, OLLAMA_TEXT_MODEL, MAX_DISTANCE_THRESHOLD, TOP_K_CHUNKS

class ManualyRAGEngine:
    def __init__(
        self, 
        vector_store: ManualVectorStore = None,
        model_name: str = OLLAMA_TEXT_MODEL,
        base_url: str = OLLAMA_BASE_URL
    ):
        self.vector_store = vector_store or ManualVectorStore()
        self.model_name = model_name
        self.api_url = f"{base_url}/api/generate"

    def _call_llm(self, prompt: str) -> str:
        """Sends prompt to local Ollama model."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            }
        }
        try:
            res = requests.post(self.api_url, json=payload, timeout=120)
            res.raise_for_status()
            return res.json().get("response", "").strip()
        except Exception as e:
            return f"Error communicating with local LLM: {e}"

    def answer_question(self, query: str, user_id: str = "default_user", doc_id: str = None) -> Dict[str, Any]:
        """
        Guarded RAG pipeline that enforces strict grounding and zero hallucination.
        """
        # 1. Retrieve top chunks
        search_results = self.vector_store.search(query=query, user_id=user_id, top_k=TOP_K_CHUNKS)
        
        documents = search_results["documents"][0] if search_results["documents"] else []
        metadatas = search_results["metadatas"][0] if search_results["metadatas"] else []
        distances = search_results["distances"][0] if search_results["distances"] else []

        # If no documents exist or best match is too distant
        if not documents:
            return self._handle_fallback(query)

        best_distance = distances[0]
        
        # Build context from retrieved chunks
        context_blocks = []
        citations = []
        evidence_images = []

        for doc_text, meta in zip(documents, metadatas):
            context_blocks.append(doc_text)
            citations.append({
                "document": meta.get("doc_name", "Manual"),
                "page": meta.get("page_number"),
                "type": meta.get("chunk_type")
            })
            if meta.get("image_path") and meta.get("image_path") not in evidence_images:
                evidence_images.append(meta["image_path"])

        context_str = "\n\n---\n\n".join(context_blocks)

        # 2. Strict Grounding Prompt
        grounded_prompt = (
            f"You are Manualy, an expert technical assistant grounded ONLY in the provided manual context.\n"
            f"Rules:\n"
            f"1. Answer the user's question using ONLY the provided manual context below.\n"
            f"2. Cite the exact page number(s) where information was found.\n"
            f"3. If the context does NOT contain enough information to answer the question, you MUST respond EXACTLY with the token: 'OUT_OF_SCOPE'. Do not invent answers.\n\n"
            f"CONTEXT FROM MANUAL:\n{context_str}\n\n"
            f"USER QUESTION: {query}\n\n"
            f"ANSWER:"
        )

        llm_response = self._call_llm(grounded_prompt)

        # 3. Check for Out-of-Scope trigger or weak distance threshold
        if "OUT_OF_SCOPE" in llm_response or best_distance > 0.85:
            return self._handle_fallback(query)

        return {
            "status": "in_scope",
            "answer": llm_response,
            "citations": citations[:2],
            "visual_evidence": evidence_images[:2],
            "best_distance": best_distance
        }

    def _handle_fallback(self, query: str) -> Dict[str, Any]:
        """Generates general AI advice when manual does not cover the topic."""
        fallback_prompt = (
            f"The user asked: '{query}'. This is NOT covered in their technical manual.\n"
            f"Provide a helpful, friendly, and practical general AI suggestion or advice in 2-3 sentences."
        )
        suggestion = self._call_llm(fallback_prompt)
        
        formatted_answer = (
            f"**The uploaded manual does not speak of this!**\n\n"
            f"**Suggestion:** {suggestion}"
        )

        return {
            "status": "out_of_scope",
            "answer": formatted_answer,
            "citations": [],
            "visual_evidence": [],
            "best_distance": None
        }