import requests
from typing import Dict, Any, List
from src.retrieval.vector_store import ManualVectorStore
from src.config import OLLAMA_BASE_URL, OLLAMA_TEXT_MODEL, TOP_K_CHUNKS

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
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 400
            }
        }
        try:
            res = requests.post(self.api_url, json=payload, timeout=180)
            res.raise_for_status()
            return res.json().get("response", "").strip()
        except requests.exceptions.Timeout:
            return "Inference timed out. Please try again."
        except Exception as e:
            return f"LLM Error: {e}"

    def answer_question(self, query: str, user_id: str = "demo_user") -> Dict[str, Any]:
        search_results = self.vector_store.search(query=query, user_id=user_id, top_k=6)
        
        documents = search_results["documents"][0] if search_results["documents"] else []
        metadatas = search_results["metadatas"][0] if search_results["metadatas"] else []
        distances = search_results["distances"][0] if search_results["distances"] else []

        if not documents:
            return self._handle_fallback(query)

        context_str = "\n\n---\n\n".join(documents)
        citations = []
        evidence_images = []

        visual_keywords = ["diagram", "figure", "schematic", "drawing", "picture", "image", "look like", "where is", "show", "wiring", "pinout"]
        user_wants_visual = any(kw in query.lower() for kw in visual_keywords)

        for meta in metadatas:
            citations.append({
                "document": meta.get("doc_name"),
                "page": meta.get("page_number"),
                "type": meta.get("chunk_type")
            })
            if (meta.get("chunk_type") == "diagram" or user_wants_visual) and meta.get("image_path"):
                if meta["image_path"] not in evidence_images:
                    evidence_images.append(meta["image_path"])

        grounded_prompt = (
            f"You are Manualy, an expert technical manual assistant.\n"
            f"Here is relevant context extracted from the manual (including text and diagram descriptions):\n"
            f"---------------------\n"
            f"{context_str}\n"
            f"---------------------\n"
            f"User Question: {query}\n\n"
            f"Instructions:\n"
            f"1. Carefully answer the question using facts from the context above.\n"
            f"2. Cite the specific page number(s) where the answer is found.\n"
            f"3. Only if the provided context contains zero relevant facts or references to answer the query, reply with: OUT_OF_SCOPE.\n\n"
            f"Answer:"
        )

        llm_response = self._call_llm(grounded_prompt)

        if "OUT_OF_SCOPE" in llm_response:
            return self._handle_fallback(query)

        return {
            "status": "in_scope",
            "answer": llm_response,
            "citations": citations[:2],
            "visual_evidence": evidence_images[:2],
            "best_distance": distances[0] if distances else None
        }

    def _handle_fallback(self, query: str) -> Dict[str, Any]:
        prompt = f"The user asked: '{query}'. Provide a brief 2-sentence practical recommendation clearly noting it is general advice not found in their manual."
        suggestion = self._call_llm(prompt)
        return {
            "status": "out_of_scope",
            "answer": f"**The uploaded manual does not contain this specific information.**\n\n💡 *Suggestion:* {suggestion}",
            "citations": [],
            "visual_evidence": []
        }