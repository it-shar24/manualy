import re
import requests
from typing import Dict, Any, List
from src.retrieval.vector_store import ManualVectorStore
from src.config import OLLAMA_BASE_URL, OLLAMA_TEXT_MODEL

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
                "temperature": 0.0,
                "num_predict": 500
            }
        }
        try:
            res = requests.post(self.api_url, json=payload, timeout=120)
            res.raise_for_status()
            return res.json().get("response", "").strip()
        except requests.exceptions.Timeout:
            return "Inference timed out. Please try again."
        except Exception as e:
            return f"LLM Error: {e}"

    def _resolve_numeric_ranges(self, query: str, context_str: str) -> str:
        """Helper to resolve numeric values (e.g. FSN 250) against hyphenated ranges (e.g. 201-300)."""
        numbers_in_query = re.findall(r"\b\d{2,4}\b", query)
        range_matches = re.findall(r"(\d{2,4})\s*-\s*(\d{2,4})", context_str)
        
        hints = []
        for num_str in numbers_in_query:
            num = int(num_str)
            for start_str, end_str in range_matches:
                start, end = int(start_str), int(end_str)
                if start <= num <= end:
                    hints.append(f"Note: The query value '{num}' is covered within the range '{start_str}-{end_str}'.")
        
        return "\n".join(set(hints))

    def answer_question(self, query: str, user_id: str = "demo_user", doc_id: str = None) -> Dict[str, Any]:
        search_results = self.vector_store.search(query=query, user_id=user_id, top_k=10)
        
        raw_docs = search_results["documents"][0] if search_results.get("documents") else []
        raw_metas = search_results["metadatas"][0] if search_results.get("metadatas") else []
        raw_dists = search_results["distances"][0] if search_results.get("distances") else []

        documents, metadatas, distances = [], [], []
        for d, m, dist in zip(raw_docs, raw_metas, raw_dists):
            if doc_id and m.get("doc_id") != doc_id:
                continue
            documents.append(d)
            metadatas.append(m)
            distances.append(dist)

        if not documents:
            return self._handle_fallback(query)

        context_blocks = []
        for idx, (doc, meta) in enumerate(zip(documents, metadatas)):
            context_blocks.append(
                f"[Source {idx+1}] (Page {meta.get('page_number')}, {meta.get('chunk_type')}):\n{doc}"
            )
        context_str = "\n\n".join(context_blocks)
        
        range_hints = self._resolve_numeric_ranges(query, context_str)
        hint_section = f"\nNumerical Context Hints:\n{range_hints}\n" if range_hints else ""

        prompt = f"""You are a technical manual assistant for aviation and industrial equipment.
Answer the user's question using ONLY the provided Context.{hint_section}

CRITICAL RULES:
1. Copy all SUBTASK numbers (e.g. SUBTASK 21-61-52-...), part numbers, and callouts verbatim.
2. If a query value falls within a range in the context (e.g. FSN 250 in 201-300), describe the subtasks applicable to that range.
3. If an effectivity range has multiple subtasks or differences, list all of them.
4. End your response with the exact sources used (e.g., Sources Used: [Source 1], [Source 2]).

Context:
{context_str}

User Question: {query}

Answer:"""

        llm_response = self._call_llm(prompt)

        active_citations = []
        visual_evidence = []
        for idx, meta in enumerate(metadatas):
            tag = f"Source {idx+1}"
            if tag in llm_response:
                active_citations.append({
                    "document": meta.get("doc_name"),
                    "page": meta.get("page_number"),
                    "type": meta.get("chunk_type")
                })
                if meta.get("image_path") and meta["image_path"] not in visual_evidence:
                    visual_evidence.append(meta["image_path"])

        if not active_citations and metadatas:
            active_citations.append({
                "document": metadatas[0].get("doc_name"),
                "page": metadatas[0].get("page_number"),
                "type": metadatas[0].get("chunk_type")
            })

        return {
            "status": "in_scope",
            "answer": llm_response,
            "citations": active_citations,
            "visual_evidence": visual_evidence,
            "best_distance": distances[0] if distances else None
        }

    def _handle_fallback(self, query: str) -> Dict[str, Any]:
        return {
            "status": "out_of_scope",
            "answer": "The uploaded manual does not contain this specific information.",
            "citations": [],
            "visual_evidence": []
        }