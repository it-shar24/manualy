from src.generation.rag_engine import ManualyRAGEngine

if __name__ == "__main__":
    engine = ManualyRAGEngine()
    user_id = "user_123"

    print("\n" + "="*60)
    print("TEST 1: In-Scope Question (Present in Manual)")
    print("="*60)
    q1 = "What is the part number for the charcoal filter cartridge, and what page is it on?"
    print(f"Question: {q1}\n")
    res1 = engine.answer_question(query=q1, user_id=user_id)
    print(f"Status: {res1['status']}")
    print(f"Answer:\n{res1['answer']}")
    print(f"\nVisual Evidence: {res1['visual_evidence']}")

    print("\n" + "="*60)
    print("TEST 2: Out-of-Scope Question (NOT in Manual)")
    print("="*60)
    q2 = "Can I put milk directly inside the water tank to make hot chocolate?"
    print(f"Question: {q2}\n")
    res2 = engine.answer_question(query=q2, user_id=user_id)
    print(f"Status: {res2['status']}")
    print(f"Answer:\n{res2['answer']}")