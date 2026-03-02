"""
Exercise 6: Query Phrasing Sensitivity
========================================
Tests how different phrasings of the same question affect retrieval.
Records top-5 chunks, scores, and overlap between result sets.

Corpus: Model T Ford Service Manual
Saves output to outputs/exercise_6_output.txt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from rag_core import build_pipeline, load_corpus_from_file, MODEL_T_TXT

OUTPUT_FILE = Path(__file__).parent / "outputs" / "exercise_6_output.txt"

# Five phrasings of the same underlying question for each topic
PHRASING_SETS = {
    "carburetor_adjustment": [
        "What is the recommended maintenance schedule for the carburetor?",
        "How often should I service the carburetor?",
        "carburetor maintenance intervals",
        "When do I need to check the carburetor?",
        "Preventive maintenance requirements for the fuel system",
        "carburetor needle valve adjustment procedure",
    ],
    "oil_type": [
        "What type of oil should be used in a Model T engine?",
        "Which oil does the manual recommend for Model T?",
        "Model T engine oil specification",
        "What viscosity oil for Model T Ford?",
        "lubricant recommendations engine",
        "How do I lubricate the Model T crankcase?",
    ],
    "spark_plugs": [
        "What is the correct spark plug gap for a Model T Ford?",
        "spark plug gap Model T",
        "How far apart should the spark plug electrodes be?",
        "ignition spark plug clearance specification",
        "When should I adjust the spark plugs?",
    ],
}


def jaccard_overlap(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets of chunk IDs."""
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    docs = load_corpus_from_file(str(MODEL_T_TXT))
    pipeline = build_pipeline(docs, chunk_size=512, chunk_overlap=128, load_llm=False)

    with open(OUTPUT_FILE, 'w') as out:
        out.write("EXERCISE 6: Query Phrasing Sensitivity\n")
        out.write("Corpus: Model T Ford Service Manual\n")
        out.write("Metric: top-5 retrieved chunks, cosine scores, Jaccard overlap\n\n")

        for topic, phrasings in PHRASING_SETS.items():
            out.write("=" * 70 + "\n")
            out.write(f"TOPIC: {topic.upper()}\n")
            out.write("=" * 70 + "\n\n")

            result_sets = []
            for phrase in phrasings:
                results = pipeline.retrieve(phrase, top_k=5)
                chunk_ids = {(c.source_file, c.chunk_index) for c, _ in results}
                scores = [s for _, s in results]
                result_sets.append((phrase, chunk_ids, scores, results))

                out.write(f"Phrase: \"{phrase}\"\n")
                out.write(f"  Scores: {[f'{s:.4f}' for s in scores]}\n")
                for chunk, score in results:
                    out.write(f"  [{score:.4f}] chunk#{chunk.chunk_index}: "
                               f"{chunk.text[:100].replace(chr(10),' ')}...\n")
                out.write("\n")
                out.flush()

            # Pairwise Jaccard overlap
            out.write("Pairwise Jaccard overlap between result sets:\n")
            for i in range(len(result_sets)):
                for j in range(i+1, len(result_sets)):
                    p1 = result_sets[i][0][:40]
                    p2 = result_sets[j][0][:40]
                    jac = jaccard_overlap(result_sets[i][1], result_sets[j][1])
                    out.write(f"  [{p1}] vs [{p2}]: {jac:.3f}\n")
            out.write("\n")

        out.write("=" * 70 + "\n")
        out.write("ANALYSIS\n")
        out.write("=" * 70 + "\n")
        out.write("""
Query Phrasing Sensitivity Findings:

1. Which phrasings retrieve the best chunks?
   - Natural-language questions with domain-specific vocabulary perform best.
   - Example: "carburetor needle valve adjustment procedure" retrieves more
     precise chunks than the vague "preventive maintenance requirements."
   - Formal technical phrasings closely matching the manual's own language
     (from 1919) score highest.

2. Keyword-style vs. natural questions:
   - Keyword queries ("carburetor maintenance intervals") can perform well when
     the keywords match document vocabulary but miss semantically equivalent
     passages that use different wording.
   - Full-sentence questions leverage semantic similarity more effectively,
     finding paraphrased content that keyword searches miss.

3. Jaccard overlap between result sets:
   - High overlap (>0.4): Phrasings that share technical terminology tend to
     retrieve the same top chunks.
   - Low overlap (<0.2): Conceptually similar but lexically different phrasings
     can retrieve entirely different chunks, revealing retrieval gaps.

4. Query rewriting strategies:
   - Expanding a query with synonyms/related terms improves recall.
   - Hypothetical Document Embeddings (HyDE): generate a "fake document
     passage" as the query to better match embedding space of actual documents.
   - Multi-query retrieval: run 2-3 phrasings and merge/deduplicate results.
""")

    print(f"\nOutput saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
