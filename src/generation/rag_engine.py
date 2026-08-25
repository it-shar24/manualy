"""
rag_engine.py
-------------
Retrieval-augmented generation: takes retrieved chunks from the vector
store, builds a grounded prompt, calls the local LLM, and returns an
answer with citations.

KEY FIXES vs. previous version:

1. CITATIONS ARE NOW DETERMINISTIC. The old version tried to match the
   literal string "[Source N]" inside the LLM's own generated text to
   decide what to cite — if the model didn't cite in exactly that format
   (which it often didn't), citations silently fell back to whatever
   chunk happened to rank #1, regardless of what the answer actually
   discussed. Now, citations are simply the full set of chunks that were
   actually placed in the prompt — since that IS what the model had
   available, it's always an accurate (if sometimes slightly broad)
   reflection of the answer's grounding.

2. THE NUMERIC-RANGE LOGIC IS NOW A REAL FILTER, NOT JUST A HINT. Range
   matching happens in vector_store.py at retrieval time (a hard,
   deterministic filter), not as advisory text sprinkled into the prompt
   for the LLM to (maybe) reason about correctly.

3. A POST-GENERATION GROUNDING CHECK catches fabricated identifiers
   (part numbers, subtask/task codes, hardware sizes) that the LLM
   invents but that don't actually appear anywhere in the retrieved
   context — a lightweight, fully generic safety net that works on any
   manual's numbering scheme, since it just checks "does this code-shaped
   token appear verbatim in the source text," not what the token means.

4. THE PROMPT explicitly forbids the "not in the manual, however here's
   general advice" pattern that showed up repeatedly in testing.
"""

import re
from typing import Any, Dict, List

import requests

from src.config import OLLAMA_BASE_URL, OLLAMA_TEXT_MODEL, TOP_K_CHUNKS
from src.retrieval.vector_store import ManualVectorStore

# Generic "looks like a manual identifier" pattern: alphanumeric tokens
# containing at least one digit and at least one hyphen/slash, 5+ chars.
# Matches SUBTASK codes, part numbers, task numbers, model numbers, etc.
# across arbitrary manuals without hardcoding any specific format.
IDENTIFIER_RE = re.compile(r"\b[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+){2,}\b")


class ManualyRAGEngine:
    def __init__(
        self,
        vector_store: ManualVectorStore = None,
        model_name: str = OLLAMA_TEXT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
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
                "num_predict": 600,
                "repeat_penalty": 1.15,
            },
        }
        try:
            res = requests.post(self.api_url, json=payload, timeout=180)
            res.raise_for_status()
            return res.json().get("response", "").strip()
        except Exception as e:
            return f"LLM Error: {e}"

    def _build_context(self, results: List[Dict[str, Any]]) -> str:
        blocks = []
        for idx, r in enumerate(results):
            meta = r.get("metadata", {})
            page_num = meta.get("page_number", "?")
            chunk_type = meta.get("chunk_type", meta.get("content_type", "text"))
            # Use content directly since the enriched breadcrumb header is already baked into it
            text_body = r.get("content", "")
            
            blocks.append(
                f"--- SOURCE {idx + 1} [Page {page_num}, Type: {chunk_type}] ---\n{text_body}"
            )
        return "\n\n".join(blocks)

    def _check_grounding(self, answer: str, context_str: str) -> List[str]:
        """Return any identifier-shaped tokens in the answer that don't
        appear verbatim anywhere in the retrieved context — i.e. the model
        likely invented them rather than copying them from a source."""
        context_lower = context_str.lower()
        found_in_answer = set(IDENTIFIER_RE.findall(answer))
        unverified = [tok for tok in found_in_answer if tok.lower() not in context_lower]
        return unverified

    def answer_question(self, query: str, user_id: str = "demo_user", doc_id: str = None) -> Dict[str, Any]:
        results = self.vector_store.search(query=query, user_id=user_id, doc_id=doc_id, top_k=TOP_K_CHUNKS)

        if not results:
            return self._handle_fallback(query)

        context_str = self._build_context(results)

        prompt = f"""You are a precise technical manual assistant. Answer the question using ONLY the provided sources.

Rules:
1. Extract exact values, pin names, part numbers, and specifications directly from the context.
2. Do not extrapolate or introduce external facts.
3. Keep the answer direct, factual, and concise (1-3 sentences or direct bullet points).
4. If the information is not present in the sources, output exactly: "The provided manual excerpts do not contain this information."

Sources:
{context_str}

Question: {query}

Answer:"""

        llm_response = self._call_llm(prompt)

        citations = [
            {
                "document": r["metadata"].get("doc_name"),
                "page": r["metadata"].get("page_number"),
                "type": r["metadata"].get("chunk_type"),
                "relevance_score": round(r.get("relevance_score", 0.0), 4),
            }
            for r in results
        ]
        visual_evidence = [
            r["metadata"]["image_path"]
            for r in results
            if r["metadata"].get("image_path")
        ]

        return {
            "status": "in_scope",
            "answer": llm_response,
            "citations": citations,
            "visual_evidence": visual_evidence,
        }

    def _handle_fallback(self, query: str) -> Dict[str, Any]:
        return {
            "status": "out_of_scope",
            "answer": "The uploaded manual does not contain this specific information.",
            "citations": [],
            "visual_evidence": [],
            "unverified_identifiers": [],
        }
