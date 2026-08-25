import json
from pathlib import Path
from typing import Dict, Any, List
import requests
from src.config import OLLAMA_BASE_URL, OLLAMA_TEXT_MODEL
from src.ingestion.pdf_parser import PDFManualParser
from src.ingestion.chunker import HybridManualChunker
from src.retrieval.vector_store import ManualVectorStore
from src.generation.rag_engine import ManualyRAGEngine

def evaluate_faithfulness_and_correctness(query: str, ground_truth: str, generated_answer: str, context: str) -> Dict[str, float]:
    """Uses a lightweight LLM judge call to evaluate Faithfulness and Answer Correctness."""
    eval_prompt = f"""You are an objective evaluator for a technical manual RAG system.
Evaluate the Generated Answer based on the Context and Ground Truth.

Context:
{context}

Question: {query}
Ground Truth Answer: {ground_truth}
Generated Answer: {generated_answer}

Score the following two metrics between 0.0 and 1.0:
1. FAITHFULNESS: Is every claim in the generated answer directly supported by the context without hallucination? (1.0 = fully supported, 0.0 = completely hallucinated)
2. CORRECTNESS: Does the generated answer accurately provide the information described in the ground truth? (1.0 = fully correct, 0.0 = wrong)

Output your response in EXACTLY this JSON format:
{{"faithfulness": 1.0, "correctness": 1.0}}
"""
    payload = {
        "model": OLLAMA_TEXT_MODEL,
        "prompt": eval_prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 80}
    }
    try:
        res = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=30)
        if res.status_code == 200:
            text = res.json().get("response", "").strip()
            # Extract JSON block
            import re
            match = re.search(r"\{.*?\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return {
                    "faithfulness": float(data.get("faithfulness", 0.0)),
                    "correctness": float(data.get("correctness", 0.0))
                }
    except Exception:
        pass
    return {"faithfulness": 0.0, "correctness": 0.0}


def calculate_metrics():
    candidate_paths = [
        Path("data/uploads/Ardino Manual.pdf"),
        Path("data/uploads/Arduino Manual.pdf"),
        Path("data/Ardino Manual.pdf"),
        Path("data/Arduino Manual.pdf"),
    ]
    manual_path = next((p for p in candidate_paths if p.exists()), None)
    if not manual_path:
        all_pdfs = list(Path("data").glob("**/*.pdf"))
        if not all_pdfs:
            raise FileNotFoundError("No PDF file found in data/ or data/uploads/")
        manual_path = all_pdfs[0]

    print(f"--- 1. Ingesting Manual: {manual_path.name} ---")
    parser = PDFManualParser(str(manual_path))
    parsed_doc = parser.extract()
    
    print(f"--- 2. Creating Context-Enriched Chunks ---")
    chunker = HybridManualChunker(max_section_chars=2200)
    chunks = chunker.create_chunks(parsed_doc)
    
    print(f"--- 3. Indexing into Vector Store ---")
    store = ManualVectorStore()
    store.index_chunks(chunks)
    rag_engine = ManualyRAGEngine(vector_store=store)

    with open("eval_dataset.json", "r") as f:
        dataset = json.load(f)

    total_queries = len(dataset)
    recall_hits = 0
    rr_scores = []
    context_precisions = []
    faithfulness_scores = []
    correctness_scores = []
    
    print(f"\n--- 4. Benchmarking {total_queries} Queries (Retrieval + Generation) ---")
    for item in dataset:
        qid = item["id"]
        query = item["query"]
        target_page = item["target_page"]
        entities = item.get("key_entities", [])
        ground_truth = item.get("ground_truth", "")

        # 1. Retrieval
        retrieved = store.search(query=query, top_k=3)
        retrieved_pages = [r["metadata"].get("page_number") for r in retrieved]
        
        hit = target_page in retrieved_pages
        if hit:
            recall_hits += 1
            rank = retrieved_pages.index(target_page) + 1
            rr_scores.append(1.0 / rank)
        else:
            rr_scores.append(0.0)

        full_context_text = " ".join([r["content"] for r in retrieved])
        matched_entities = sum(1 for e in entities if e.lower() in full_context_text.lower())
        precision = matched_entities / len(entities) if entities else 0.0
        context_precisions.append(precision)

        # 2. Generation & End-to-End Evaluation
        rag_response = rag_engine.answer_question(query)
        gen_answer = rag_response.get("answer", "")
        
        eval_res = evaluate_faithfulness_and_correctness(
            query=query,
            ground_truth=ground_truth,
            generated_answer=gen_answer,
            context=full_context_text
        )
        faithfulness_scores.append(eval_res["faithfulness"])
        correctness_scores.append(eval_res["correctness"])

        print(f"[{qid}] Target: p{target_page} | Hit: {hit} | MRR: {rr_scores[-1]:.2f} | Faithfulness: {eval_res['faithfulness']:.2f} | Correctness: {eval_res['correctness']:.2f}")

    avg_recall = (recall_hits / total_queries) * 100
    avg_mrr = sum(rr_scores) / total_queries
    avg_context_precision = (sum(context_precisions) / total_queries) * 100
    avg_faithfulness = (sum(faithfulness_scores) / total_queries) * 100
    avg_correctness = (sum(correctness_scores) / total_queries) * 100

    print("\n" + "=" * 50)
    print("      END-TO-END RAG BENCHMARK REPORT")
    print("=" * 50)
    print(f"Total Benchmark Queries:     {total_queries}")
    print(f"Retrieval Recall@3:          {avg_recall:.1f}%")
    print(f"Mean Reciprocal Rank (MRR):  {avg_mrr:.3f}")
    print(f"Context Precision:           {avg_context_precision:.1f}%")
    print("-" * 50)
    print(f"Answer Faithfulness:         {avg_faithfulness:.1f}%")
    print(f"Answer Correctness:          {avg_correctness:.1f}%")
    print("=" * 50)

if __name__ == "__main__":
    calculate_metrics()