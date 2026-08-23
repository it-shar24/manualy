from src.ingestion.pdf_parser import PDFManualParser
from src.ingestion.chunker import HybridManualChunker
from src.retrieval.vector_store import ManualVectorStore
from src.generation.rag_engine import ManualyRAGEngine

pdf_path = "data/uploads/AMM.pdf"

print("--- 1. Ingesting AMM.pdf ---")
parser = PDFManualParser(pdf_path)
parsed = parser.extract()
print(f"Extracted {parsed['total_pages']} pages, {parsed['total_images_extracted']} figures.")
for p in parsed["pages"]:
    print(f"Page {p['page_number']}: Text length = {len(p['text'])} chars")

print("\n--- 2. Building Chunks ---")
chunker = HybridManualChunker()
chunks = chunker.create_chunks(parsed, user_id="demo_user")
print(f"Total Chunks Created: {len(chunks)} (Types: {[c['metadata']['chunk_type'] for c in chunks]})")

print("\n--- 3. Indexing to ChromaDB ---")
store = ManualVectorStore()
store.index_chunks(chunks)

print("\n--- 4. Running Diagnostic Queries ---")
rag = ManualyRAGEngine(vector_store=store)

print("\n========== Q1 ==========")
res_q1 = rag.answer_question("What is the subtask number and procedure for FSN 250?", user_id="demo_user", doc_id="AMM")
print("ANSWER:\n", res_q1["answer"])
print("CITATIONS:", res_q1["citations"])

print("\n========== Q2 ==========")
res_q2 = rag.answer_question("What is the difference between LH and RH bolts?", user_id="demo_user", doc_id="AMM")
print("ANSWER:\n", res_q2["answer"])
print("CITATIONS:", res_q2["citations"])