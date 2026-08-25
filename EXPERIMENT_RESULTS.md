# Technical Manual RAG Benchmark & Strategy Evaluation

## 1. Executive Summary
Evaluation of four chunking strategies on the `Arduino UNO R3 User Manual` (26 pages) across a 20-question ground-truth benchmark.

**Retrieval Pipeline:**
* First Stage: Dense Vector Embedding (MiniLM) + Sparse BM25 Keyword Search with Reciprocal Rank Fusion ($k=25$).
* Second Stage: Cross-Encoder Reranker (`ms-marco-TinyBERT`) returning the top-$3$ candidates to `Llama-3.2:3b`.

---

## 2. Chunking Strategy Comparison Table

| Strategy | Description | Total Chunks | Recall@3 | MRR | Context Precision |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Fixed-Size Overlap** | Sliding window (2,000 chars / 400 overlap) | 29 | **85.0%** | **0.767** | **80.0%** |
| **2. Structure-Aware** | Headings, section IDs & table bounds | *Pending* | — | — | — |
| **3. Parent-Child + Window** | Small child chunks with full parent context | *Pending* | — | — | — |
| **4. Semantic Chunking** | Embedding cosine distance breakpoint splits | *Pending* | — | — | — |

---

## 3. Strategy Benchmark Reports

==================================================
      STRATEGY 1 (FIXED-SIZE) BENCHMARK REPORT
==================================================
Total Benchmark Queries:     20
Retrieval Recall@3:          85.0%
Mean Reciprocal Rank (MRR):  0.767
Context Precision:           80.0%
==================================================


==================================================
      STRATEGY 2 (STRUCTURE-AWARE) BENCHMARK REPORT
==================================================
Total Benchmark Queries:     20
Retrieval Recall@3:          95.0%
Mean Reciprocal Rank (MRR):  0.858
Context Precision:           76.2%
==================================================


==================================================
      STRATEGY 3 (PARENT-CHILD) BENCHMARK REPORT
==================================================
Total Benchmark Queries:     20
Retrieval Recall@3:          85.0%
Mean Reciprocal Rank (MRR):  0.800
Context Precision:           80.7%
==================================================



==================================================
      STRATEGY 4 (SEMANTIC CHUNKING) BENCHMARK REPORT
==================================================
Total Benchmark Queries:     20
Retrieval Recall@3:          75.0%
Mean Reciprocal Rank (MRR):  0.725
Context Precision:           72.5%
==================================================



## 3. Production Benchmark Report

==================================================
  STRATEGY 2 (ENRICHED STRUCTURE-AWARE) BENCHMARK REPORT
==================================================
Total Benchmark Queries:     20
Retrieval Recall@3:          95.0%
Mean Reciprocal Rank (MRR):  0.858
Context Precision:           76.2%
==================================================


##  End-to-End Generation Benchmark (Strategy 2 + Llama-3.2:3b)

==================================================
      END-TO-END RAG BENCHMARK REPORT
==================================================
Total Benchmark Queries:     20
Retrieval Recall@3:          95.0%
Mean Reciprocal Rank (MRR):  0.858
Context Precision:           76.2%
--------------------------------------------------
Answer Faithfulness:         45.0%
Answer Correctness:          50.0%
==================================================

# Generation Prompt Grounding Calibration

==================================================
      END-TO-END RAG BENCHMARK REPORT
==================================================
Total Benchmark Queries:     20
Retrieval Recall@3:          95.0%
Mean Reciprocal Rank (MRR):  0.858
Context Precision:           76.2%
--------------------------------------------------
Answer Faithfulness:         75.0%
Answer Correctness:          80.0%
==================================================