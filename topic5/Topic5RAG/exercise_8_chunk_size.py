"""
Exercise 8: Chunk Size Experiment
====================================
Tests how chunk size affects retrieval precision and answer quality.
Sizes: 128, 512, 2048 characters (overlap fixed at chunk_size // 4).

NOTE: Rebuilds the index for each chunk size. Recommended: GPU or MPS.

Corpus: Model T Ford Service Manual
Saves output to outputs/exercise_8_output.txt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
import time
from rag_core import (
    RAGPipeline, load_corpus_from_file,
    SentenceTransformer, EMBEDDING_MODEL_NAME, MODEL_T_TXT, get_device,
    LLM_MODEL_NAME
)

OUTPUT_FILE = Path(__file__).parent / "outputs" / "exercise_8_output.txt"

CHUNK_SIZES = [128, 512, 2048]

QUERIES = [
    "How do I adjust the carburetor on a Model T?",
    "What is the correct spark plug gap for a Model T Ford?",
    "How do I fix a slipping transmission band?",
    "What oil should I use in a Model T engine?",
    "How do I time the ignition system?",
]


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    docs = load_corpus_from_file(str(MODEL_T_TXT))

    device, dtype = get_device()
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading LLM: {LLM_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    if device == 'mps':
        llm = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME, dtype=dtype, trust_remote_code=True
        ).to(device)
    else:
        llm = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME, dtype=dtype, trust_remote_code=True)
    llm.eval()

    pipeline = RAGPipeline(device=device, dtype=dtype,
                           embed_model=embed_model,
                           llm_model=llm, llm_tokenizer=tokenizer)

    with open(OUTPUT_FILE, 'w') as out:
        out.write("EXERCISE 8: Chunk Size Experiment\n")
        out.write("Corpus: Model T Ford Service Manual\n")
        out.write("Overlap = chunk_size // 4 for each configuration\n\n")

        size_stats = {}

        for cs in CHUNK_SIZES:
            overlap = cs // 4
            out.write("=" * 70 + "\n")
            out.write(f"CHUNK SIZE = {cs}  (overlap={overlap})\n")
            out.write("=" * 70 + "\n")

            t0 = time.time()
            pipeline.build_index(docs, chunk_size=cs, chunk_overlap=overlap)
            build_time = time.time() - t0
            n = len(pipeline.chunks)
            size_stats[cs] = (n, build_time)

            out.write(f"Chunks: {n}  |  Build time: {build_time:.1f}s\n\n")

            for q in QUERIES:
                out.write(f"Query: {q}\n")
                ans, results = pipeline.rag_query(q, top_k=5)
                scores = [f"{s:.4f}" for _, s in results]
                out.write(f"Scores: {scores}\n")
                out.write(f"Answer:\n{ans}\n\n")
                out.flush()

        out.write("=" * 70 + "\n")
        out.write("SUMMARY TABLE\n")
        out.write("=" * 70 + "\n")
        out.write(f"{'Chunk Size':>12} {'# Chunks':>10} {'Build Time':>12}\n")
        for cs, (n, bt) in size_stats.items():
            out.write(f"{cs:>12} {n:>10} {bt:>11.1f}s\n")

        out.write("\n" + "=" * 70 + "\n")
        out.write("ANALYSIS\n")
        out.write("=" * 70 + "\n")
        out.write("""
Chunk Size Effect:

chunk_size=128 (very small):
  + High retrieval precision: matches the exact sentence containing the answer.
  - Low completeness: procedural answers (multi-step) require many chunks;
    k=5 may not cover the full procedure.
  - Index is largest (most chunks); embedded context is granular.
  - Works well for: single-fact lookups (gap = X, oil = Y).

chunk_size=512 (medium — default):
  + Good balance of precision and completeness.
  + Enough context to carry a full procedure or paragraph.
  - Occasional irrelevant sentences within a chunk add noise.
  - Sweet spot for most RAG applications.

chunk_size=2048 (very large):
  + Excellent completeness: one chunk covers an entire section.
  - Low precision: retrieval matches at topic level, not sentence level;
    answer may be buried in a large context block.
  - Fewer total chunks → coarser index; may miss answers in "wrong section."
  - Good for synthesis tasks; bad for targeted factual retrieval.

Does optimal size depend on question type?
  - Yes. Fact retrieval: small (128-256). Procedure recall: medium (512).
  - Cross-section synthesis: large (1024-2048) or high k.

Sweet spot for this corpus: 512 characters.
""")

    print(f"\nOutput saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
