import json
from pathlib import Path
from src.ingestion.pdf_parser import PDFManualParser
from src.ingestion.chunker import HybridManualChunker
from src.retrieval.vector_store import ManualVectorStore
from src.generation.rag_engine import ManualyRAGEngine

def calculate_metrics():
    # Robust path resolution to handle multiple upload folders or filename variations
    candidate_paths = [
        Path("data/uploads/Ardino Manual.pdf"),
    ]
    manual_path = next((p for p in candidate_paths if p.exists()), None)
    
    if not manual_path:
        # Fallback to any PDF in data/ or data/uploads/
        all_pdfs = list(Path("data").glob("**/*.pdf"))
        if not all_pdfs:
            raise FileNotFoundError("No PDF file found in data/ or data/uploads/")
        manual_path = all_pdfs[0]

    print(f"--- 1. Ingesting Manual: {manual_path.name} ---")
    parser = PDFManualParser(str(manual_path))
    parsed_doc = parser.extract()
    
    print(f"--- 2. Creating Context-Enriched Structure-Aware Chunks (Strategy 2) ---")
    chunker = HybridManualChunker(max_section_chars=2500)
    chunks = chunker.create_chunks(parsed_doc)
    print(f"Total Chunks Created: {len(chunks)}")
    
    print(f"--- 3. Indexing to Vector Store + Cross-Encoder ---")
    store = ManualVectorStore()
    store.index_chunks(chunks)
    rag_engine = ManualyRAGEngine(vector_store=store)

    with open("eval_dataset.json", "r") as f:
        dataset = json.load(f)

    total_queries = len(dataset)
    recall_hits = 0
    rr_scores = []
    context_precisions = []
    
    print(f"\n--- 4. Benchmarking 20 Ground Truth Queries ---")
    for item in dataset:
        qid = item["id"]
        query = item["query"]
        target_page = item["target_page"]
        entities = item["key_entities"]

        # Vector Search (Top-25) -> Cross-Encoder Reranker (Top-3)
        retrieved = store.search(query=query, top_k=3)
        retrieved_pages = [r["metadata"].get("page_number") for r in retrieved]
        
        # 1. Recall@3 & MRR
        hit = target_page in retrieved_pages
        if hit:
            recall_hits += 1
            rank = retrieved_pages.index(target_page) + 1
            rr_scores.append(1.0 / rank)
        else:
            rr_scores.append(0.0)

        # 2. Context Precision across the retrieved enriched chunks
        full_context_text = " ".join([r["content"].lower() for r in retrieved])
        matched_entities = sum(1 for e in entities if e.lower() in full_context_text)
        precision = matched_entities / len(entities) if entities else 0.0
        context_precisions.append(precision)

        print(f"[{qid}] Target Page: {target_page} | Retrieved: {retrieved_pages} | Hit: {hit} | MRR: {rr_scores[-1]:.2f}")

    avg_recall = (recall_hits / total_queries) * 100
    avg_mrr = sum(rr_scores) / total_queries
    avg_context_precision = (sum(context_precisions) / total_queries) * 100

    print("\n" + "=" * 50)
    print("  STRATEGY 2 (ENRICHED STRUCTURE-AWARE) BENCHMARK REPORT")
    print("=" * 50)
    print(f"Total Benchmark Queries:     {total_queries}")
    print(f"Retrieval Recall@3:          {avg_recall:.1f}%")
    print(f"Mean Reciprocal Rank (MRR):  {avg_mrr:.3f}")
    print(f"Context Precision:           {avg_context_precision:.1f}%")
    print("=" * 50)

if __name__ == "__main__":
    calculate_metrics()