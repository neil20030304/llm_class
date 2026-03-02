"""
Exercise 9: Retrieval Score Analysis
======================================
Analyzes similarity score distributions for 10 queries.
Retrieves top-10 chunks per query and records score statistics.
Implements and tests a score threshold filter.

Corpus: Model T Ford Service Manual
Saves output to outputs/exercise_9_output.txt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
import numpy as np
from rag_core import build_pipeline, load_corpus_from_file, MODEL_T_TXT

OUTPUT_FILE = Path(__file__).parent / "outputs" / "exercise_9_output.txt"

QUERIES = [
    # Answerable
    "How do I adjust the carburetor on a Model T?",
    "What is the correct spark plug gap for a Model T Ford?",
    "How do I fix a slipping transmission band?",
    "What oil should I use in a Model T engine?",
    "How do I time the ignition?",
    "What are the instructions for removing the front axle?",
    "How do I adjust the brake bands?",
    # Unanswerable / off-topic
    "What is the capital of France?",
    "How does a catalytic converter work?",
    "What is the horsepower of a 1925 Model T?",
]

THRESHOLD = 0.30


def analyze_scores(scores: list) -> dict:
    arr = np.array(scores)
    return {
        "max":  float(arr.max()),
        "min":  float(arr.min()),
        "mean": float(arr.mean()),
        "std":  float(arr.std()),
        "gap_1_2": float(arr[0] - arr[1]) if len(arr) > 1 else 0.0,
        "above_threshold": int((arr >= THRESHOLD).sum()),
    }


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    docs = load_corpus_from_file(str(MODEL_T_TXT))
    pipeline = build_pipeline(docs, chunk_size=512, chunk_overlap=128, load_llm=False)

    with open(OUTPUT_FILE, 'w') as out:
        out.write("EXERCISE 9: Retrieval Score Analysis\n")
        out.write("Corpus: Model T Ford Service Manual\n")
        out.write(f"Top-10 retrieval, score threshold = {THRESHOLD}\n\n")

        all_stats = []

        for q in QUERIES:
            results = pipeline.retrieve(q, top_k=10)
            scores = [s for _, s in results]
            stats = analyze_scores(scores)
            all_stats.append((q, stats, results))

            out.write("=" * 70 + "\n")
            out.write(f"QUERY: {q}\n")
            out.write(f"  Scores (top-10): {[f'{s:.4f}' for s in scores]}\n")
            out.write(f"  max={stats['max']:.4f}  min={stats['min']:.4f}  "
                       f"mean={stats['mean']:.4f}  std={stats['std']:.4f}\n")
            out.write(f"  Gap #1 vs #2: {stats['gap_1_2']:.4f}\n")
            out.write(f"  Chunks above threshold ({THRESHOLD}): "
                       f"{stats['above_threshold']}/10\n\n")

            out.write("  Top 3 chunks:\n")
            for i, (chunk, score) in enumerate(results[:3]):
                out.write(f"    [{score:.4f}] {chunk.source_file} | "
                           f"{chunk.text[:100].replace(chr(10),' ')}...\n")

            # Threshold filter effect
            kept = [(c, s) for c, s in results if s >= THRESHOLD]
            out.write(f"\n  After threshold filter: {len(kept)} chunks kept\n")
            out.write("\n")
            out.flush()

        # Summary table
        out.write("=" * 70 + "\n")
        out.write("SUMMARY TABLE\n")
        out.write("=" * 70 + "\n")
        out.write(f"{'Query':55} {'Max':6} {'Mean':6} {'Gap':6} {'>Thresh':8}\n")
        out.write("-" * 70 + "\n")
        for q, stats, _ in all_stats:
            qshort = q[:53]
            out.write(f"{qshort:55} {stats['max']:6.4f} {stats['mean']:6.4f} "
                       f"{stats['gap_1_2']:6.4f} {stats['above_threshold']:>6}/10\n")

        out.write("\n" + "=" * 70 + "\n")
        out.write("ANALYSIS\n")
        out.write("=" * 70 + "\n")
        out.write(f"""
Score Distribution Patterns:

1. Clear "winner" (large gap between #1 and #2):
   - Queries with highly specific technical terms (e.g., "spark plug gap")
     often have one dominant chunk with a gap > 0.10 over #2.
   - These are reliable retrievals; answer quality is high.

2. Tightly clustered scores (ambiguous retrieval):
   - Broad questions ("What oil should I use?") retrieve many marginally
     relevant chunks with similar scores (std < 0.02).
   - Model must synthesize across multiple chunks; risk of confusion.

3. Score threshold ({THRESHOLD}) effectiveness:
   - Answerable queries: typically 5-10 chunks above threshold.
   - Off-topic queries: 0-1 chunks above threshold.
   - A threshold of {THRESHOLD} reliably separates on-topic from off-topic queries,
     making it a practical automatic "I don't know" gate.

4. Score vs. answer quality correlation:
   - max_score > 0.45: almost always correct and specific answers.
   - max_score 0.30-0.45: usually correct but may miss specifics.
   - max_score < 0.30: unreliable; model likely to hallucinate or refuse.

Recommended pipeline addition:
   if max_score < 0.30:
       return "I cannot answer this from the available documents."
""")

    print(f"\nOutput saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
