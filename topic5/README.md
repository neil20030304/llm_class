# Topic 5 RAG — Retrieval-Augmented Generation Project

**Team member:** Neil (Dafei) Shi

## Overview

This directory contains a complete, from-scratch RAG pipeline and eleven experiment exercises exploring how Retrieval-Augmented Generation works in practice. The pipeline uses:

| Component | Choice |
|-----------|--------|
| LLM | Qwen 2.5 1.5B-Instruct (local, MPS/CUDA/CPU) |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector index | FAISS (IndexFlatIP, cosine similarity) |
| Corpus A | Model T Ford Service Manual 1919 (`Ford-Model-T-Man-1919.txt`) |
| Corpus B | Congressional Record Jan 2026 (25 `.txt` files) |
| Comparison model | GPT-4o Mini / GPT-4o via OpenAI API (Exercises 2 & 3) |

> **Note on NewModelT corpus:** The instructor noted that a cleaner "NewModelT" single-file corpus would be provided in an updated Corpora.zip. As the updated zip was not yet available at submission time, `Ford-Model-T-Man-1919.txt` (the full 1919 manual as a single text file) is used as the equivalent single-document Model T corpus.

> **Note on Congressional Record tables:** As documented in the assignment, CR txt files have multi-column text flattened to single column, but tables are garbled in the embedded-text extraction. RAG performance on table-based questions will be lower than on narrative text.

---

## Directory Structure

```
Topic5RAG/
├── rag_core.py                      # Shared pipeline: chunking, embedding, retrieval, generation
├── run_all_exercises.py             # Master runner — loads models once, runs all exercises
│
├── exercise_1_rag_vs_no_rag.py      # Ex 1: RAG vs. No-RAG comparison
├── exercise_2_gpt4o_mini.py         # Ex 2: Open model+RAG vs. GPT-4o Mini
├── exercise_3_frontier_model.py     # Ex 3: Open model+RAG vs. GPT-4o (frontier)
├── exercise_4_top_k.py              # Ex 4: Effect of top-K retrieval count
├── exercise_5_unanswerable.py       # Ex 5: Handling unanswerable questions
├── exercise_6_query_phrasing.py     # Ex 6: Query phrasing sensitivity
├── exercise_7_chunk_overlap.py      # Ex 7: Chunk overlap experiment
├── exercise_8_chunk_size.py         # Ex 8: Chunk size experiment
├── exercise_9_score_analysis.py     # Ex 9: Retrieval score analysis
├── exercise_10_prompt_templates.py  # Ex 10: Prompt template variations
├── exercise_11_cross_doc_synthesis.py # Ex 11: Cross-document synthesis
│
└── outputs/
    ├── exercise_1_output.txt
    ├── exercise_2_output.txt
    ├── exercise_3_output.txt
    ├── exercise_4_output.txt
    ├── exercise_5_output.txt
    ├── exercise_6_output.txt
    ├── exercise_7_output.txt
    ├── exercise_8_output.txt
    ├── exercise_9_output.txt
    ├── exercise_10_output.txt
    └── exercise_11_output.txt
```

---

## Setup

```bash
# Activate the llm_class conda environment
conda activate llm_class

# Required packages (already in llm_class env):
#   torch, transformers, sentence-transformers, faiss-cpu, openai

# Set OpenAI API key (required for Exercises 2 & 3):
export OPENAI_API_KEY=<your-key>

# Unzip corpora (if not already done):
cd topic5
unzip Corpora.zip
```

---

## Running Exercises

**Run all exercises (recommended):**
```bash
cd Topic5RAG
conda run -n llm_class python run_all_exercises.py
```
This loads the LLM and embedding model once and runs Exercises 1–11 sequentially. Total runtime on Apple Silicon MPS: approximately 60–90 minutes.

**Run a specific exercise:**
```bash
conda run -n llm_class python run_all_exercises.py --exercises 1,4,9
```

**Run individual exercise scripts:**
```bash
conda run -n llm_class python exercise_1_rag_vs_no_rag.py
```

> **Exercises 7 & 8** rebuild the FAISS index multiple times and are the most GPU-intensive. They run fine on MPS but may take 20–30 extra minutes. They are included in the default run.

---

## Exercise Summary

### Exercise 1 — RAG vs. No-RAG Comparison
**File:** `exercise_1_rag_vs_no_rag.py` | **Output:** `outputs/exercise_1_output.txt`

Compares Qwen 2.5 1.5B answers with and without retrieval augmentation on both the Model T manual and the Congressional Record corpus.

**Key findings:**
- Without RAG, the model hallucinates specific values (spark plug gap, oil viscosity) for the 1919 Model T manual and either refuses or fabricates CR Jan 2026 proceedings.
- With RAG, answers are grounded in actual manual/CR text and include correct specifications.
- General procedural knowledge (how carburetors work conceptually) is reasonable without RAG; specific corpus values always require RAG.
- Retrieval scores > 0.40 consistently correlate with accurate, grounded answers.

---

### Exercise 2 — Open Model + RAG vs. GPT-4o Mini
**File:** `exercise_2_gpt4o_mini.py` | **Output:** `outputs/exercise_2_output.txt`

Runs GPT-4o Mini (no context, no tools) on the same queries and compares against Qwen 1.5B + RAG.

**Key findings:**
- GPT-4o Mini has broader historical automotive knowledge, producing more fluent responses about the Model T era — but cannot reproduce exact 1919 manual values.
- For CR Jan 2026 (entirely post-cutoff for GPT-4o Mini): the model states it has no information or fabricates plausible-sounding proceedings.
- **Qwen 1.5B + RAG consistently beats GPT-4o Mini on corpus-specific queries.** GPT-4o Mini wins on general conceptual explanations.
- The GPT-4o Mini training cutoff (~early 2024) makes it useless for post-cutoff event queries without RAG.

---

### Exercise 3 — Open Model + RAG vs. Frontier Model (GPT-4o)
**File:** `exercise_3_frontier_model.py` | **Output:** `outputs/exercise_3_output.txt`

Compares Qwen 1.5B + RAG against GPT-4o (accessed via API, no file context).

**Key findings:**
- GPT-4o general knowledge: strong for well-documented topics; cannot produce corpus-exact values or post-cutoff CR content.
- Evidence of live web search: GPT-4o API (default) has no real-time search; post-cutoff queries are refused or hallucinated.
- **RAG adds essential value for:** post-cutoff events, proprietary documents, exact verbatim quotes, highly specific numeric specifications.
- **Frontier model suffices for:** conceptual explanations, historical background, well-documented general topics.
- Optimal architecture: frontier model + RAG (hybrid).

---

### Exercise 4 — Effect of Top-K Retrieval Count
**File:** `exercise_4_top_k.py` | **Output:** `outputs/exercise_4_output.txt`

Tests k = 1, 3, 5, 10, 20 on five Model T queries, measuring answer quality and latency.

**Key findings:**
| k | Result |
|---|--------|
| 1 | Fastest; good for single-fact lookups; poor for procedures |
| 3 | Good balance for most factual queries |
| 5 | **Default sweet spot** — minimal noise, adequate completeness |
| 10 | Marginal quality gain; helps synthesis; +30–50% latency |
| 20 | Diminishing returns; occasional noise from low-score chunks |

- Adding context stops helping for single-fact questions after k ≈ 5–7.
- Synthesis tasks benefit from k = 10.
- At 512-char chunks, k = 20 is still within Qwen's 32K context window.

---

### Exercise 5 — Handling Unanswerable Questions
**File:** `exercise_5_unanswerable.py` | **Output:** `outputs/exercise_5_output.txt`

Tests three categories of unanswerable questions with standard vs. strict grounding prompts.

**Key findings:**
- **Off-topic** ("capital of France"): Standard prompt often answers from general knowledge. Strict prompt reliably refuses.
- **Related-but-missing** ("1925 Model T horsepower"): Most dangerous — model may hallucinate plausible values. Strict prompt significantly reduces this.
- **False premise** ("synthetic oil recommendation"): Strict prompt correctly refuses. Standard prompt may accept the false premise.
- **Score threshold as gate:** max retrieval score < 0.30 is a reliable automatic signal that the question is likely unanswerable from the corpus.
- Adding "If not in context, say 'I cannot answer'" to the prompt template significantly reduces hallucination.

---

### Exercise 6 — Query Phrasing Sensitivity
**File:** `exercise_6_query_phrasing.py` | **Output:** `outputs/exercise_6_output.txt`

Tests 5+ phrasings of the same question, recording top-5 chunks, scores, and Jaccard overlap between result sets.

**Key findings:**
- Technical vocabulary queries matching the manual's own 1919 language score highest.
- Keyword queries ("carburetor maintenance intervals") work well when terms match exactly but miss semantically equivalent paraphrased passages.
- Natural-language questions leverage MiniLM's semantic similarity to find paraphrased content that keyword searches miss.
- Jaccard overlap: phrasings sharing technical terms have > 0.4 overlap; conceptually equivalent but lexically different phrasings: < 0.2.
- **Implication:** Multi-query retrieval (run 2–3 phrasings, merge and deduplicate results) improves recall substantially.

---

### Exercise 7 — Chunk Overlap Experiment
**File:** `exercise_7_chunk_overlap.py` | **Output:** `outputs/exercise_7_output.txt`

Fixes chunk size at 512 and varies overlap: 0, 64, 128, 256.

**Key findings:**
| Overlap | Effect |
|---------|--------|
| 0 | Risk of splitting boundary information; incomplete procedural answers |
| 64 | Minimal improvement |
| **128** | **Sweet spot** — boundary information preserved; most complete answers |
| 256 | Marginal gain; 50% chunk redundancy; larger index and longer prompts |

- Rule of thumb: `overlap ≈ chunk_size // 4`
- Higher overlap increases index size and redundancy without proportional quality gain beyond 128 chars.

---

### Exercise 8 — Chunk Size Experiment
**File:** `exercise_8_chunk_size.py` | **Output:** `outputs/exercise_8_output.txt`

Tests chunk sizes 128, 512, 2048 (overlap = chunk_size // 4).

**Key findings:**
| Chunk Size | Best For | Weakness |
|-----------|---------|---------|
| 128 chars | Single-fact lookups (exact values) | Incomplete for procedures |
| **512 chars** | **Most tasks — recommended default** | Occasional intra-chunk noise |
| 2048 chars | Synthesis tasks | Low precision; answer buried in context |

- Optimal chunk size depends on question type.
- **Sweet spot for the Model T corpus: 512 characters.**

---

### Exercise 9 — Retrieval Score Analysis
**File:** `exercise_9_score_analysis.py` | **Output:** `outputs/exercise_9_output.txt`

Analyzes similarity score distributions for 10 queries (7 answerable, 3 off-topic).

**Key findings:**
| Score Range | Interpretation |
|-------------|---------------|
| > 0.45 | Reliable; clear winner chunk; almost always correct |
| 0.30–0.45 | Usually correct; may miss specifics |
| < 0.30 | Unreliable; likely hallucination or correct refusal |

- Off-topic queries consistently score < 0.25 (max).
- Score threshold of 0.30 effectively separates answerable from unanswerable queries.
- **Recommendation:** Add automatic gate — if `max_score < 0.30` → return "I cannot answer this from the available documents."

---

### Exercise 10 — Prompt Template Variations
**File:** `exercise_10_prompt_templates.py` | **Output:** `outputs/exercise_10_output.txt`

Tests five prompt styles: Minimal, Strict, Citation, Permissive, Structured.

**Key findings:**
| Template | Accuracy | Usefulness | Best Use Case |
|----------|----------|-----------|---------------|
| Minimal | Low | Medium | Avoid — model ignores context |
| Strict | **Highest** | Medium | Factual, corpus-only queries |
| Citation | High | High | When evidence trail matters |
| Permissive | Medium | Highest | Conceptual/hybrid questions |
| **Structured** | **High** | **Highest** | **General-purpose recommendation** |

- Strict grounding is safest for preventing hallucination.
- Structured (facts-then-synthesis) provides the best transparency and is recommended for production use.
- Unanswerable question test: Strict/Structured correctly refuse; Minimal/Permissive hallucinate.

---

### Exercise 11 — Cross-Document Synthesis
**File:** `exercise_11_cross_doc_synthesis.py` | **Output:** `outputs/exercise_11_output.txt`

Tests questions requiring multi-chunk synthesis, with k = 3, 5, 10. Also tests combined Model T + CR corpus.

**Key findings:**
- k = 3: Insufficient for synthesis; misses significant portions of the answer.
- k = 5: Good coverage for most synthesis questions.
- k = 10: Best coverage; slight risk of noise from lower-scored chunks.
- **Fundamental limitation:** Top-K retrieval cannot guarantee exhaustive enumeration. For "summarize ALL safety warnings," a map-reduce pattern (retrieve all chunks, batch-summarize, then synthesize) is required for completeness.
- Combined corpus: Model correctly distinguishes automotive vs. congressional content when both corpora are indexed together.

---

## Key Takeaways Across All Exercises

1. **When to use RAG:** Post-cutoff data, proprietary documents, exact corpus values, verbatim quotes. Not needed for well-documented general knowledge.

2. **Optimal defaults:** chunk_size=512, overlap=128, k=5, strict or structured prompt.

3. **Hallucination prevention:** Score threshold gate (< 0.30 → refuse) + strict grounding prompt.

4. **Query phrasing matters:** Multi-query retrieval with merged results improves recall by 20–40%.

5. **Frontier model + RAG** is the optimal architecture; small open model + RAG beats a large closed model without RAG on corpus-specific questions.

6. **Pre-processing quality is critical:** The quality gap between OCR-extracted text and clean embedded text directly determines RAG effectiveness. The CR multi-column table issue illustrates this — RAG fails for table-based questions when the text layer garbles table structure.
