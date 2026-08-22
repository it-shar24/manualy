from src.ingestion.pdf_parser import PDFManualParser
from src.ingestion.chunker import HybridManualChunker
from src.retrieval.vector_store import ManualVectorStore

if __name__ == "__main__":
    pdf_path = "data/uploads/SmartBrew_SB3000_Manual.pdf"

    print("\n--- 1. Parsing PDF ---")
    parser = PDFManualParser(pdf_path)
    parsed_doc = parser.extract()

    print("\n--- 2. Building Hybrid Chunks & Vision Descriptions ---")
    chunker = HybridManualChunker()
    chunks = chunker.create_chunks(parsed_doc, user_id="user_123")

    print(f"\nCreated {len(chunks)} total hybrid chunks.")

    print("\n--- 3. Indexing into ChromaDB ---")
    store = ManualVectorStore()
    store.index_chunks(chunks)

    print("\n--- 4. Test Search Query: 'What is the part number for the charcoal filter?' ---")
    search_results = store.search(
        query="What is the part number for the charcoal filter?", 
        user_id="user_123",
        top_k=2
    )

    print("\nSearch Results:")
    for i in range(len(search_results["documents"][0])):
        print(f"\nResult {i+1}:")
        print(f"Distance: {search_results['distances'][0][i]:.4f}")
        print(f"Metadata: {search_results['metadatas'][0][i]}")
        print(f"Snippet : {search_results['documents'][0][i][:160]}...")