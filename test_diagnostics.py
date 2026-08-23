from src.retrieval.vector_store import ManualVectorStore
from src.generation.rag_engine import ManualyRAGEngine

store = ManualVectorStore()
rag = ManualyRAGEngine(vector_store=store)

print("\n========== TESTING Q1 ==========")
res_q1 = rag.answer_question("What is the subtask number and procedure for FSN 250?", user_id="demo_user")
print("ANSWER:\n", res_q1["answer"])
print("CITATIONS:", res_q1["citations"])

print("\n========== TESTING Q2 ==========")
res_q2 = rag.answer_question("What is the difference between LH and RH bolts?", user_id="demo_user")
print("ANSWER:\n", res_q2["answer"])
print("CITATIONS:", res_q2["citations"])