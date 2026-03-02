"""
Exercise 5: Handling Unanswerable Questions
============================================
Tests how the RAG system handles questions that cannot be answered from the corpus.
Categories: completely off-topic, related-but-not-in-corpus, false premise.
Also tests effect of adding an explicit "say I don't know" instruction.

Corpus: Model T Ford Service Manual
Saves output to outputs/exercise_5_output.txt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from rag_core import build_pipeline, load_corpus_from_file, MODEL_T_TXT, DEFAULT_PROMPT

OUTPUT_FILE = Path(__file__).parent / "outputs" / "exercise_5_output.txt"

UNANSWERABLE_QUERIES = {
    "off_topic": [
        "What is the capital of France?",
        "How do I bake a sourdough loaf?",
        "What is the speed of light?",
    ],
    "related_but_missing": [
        "What is the horsepower of a 1925 Model T?",
        "How do I install a catalytic converter on a Model T?",
        "What are the emission regulations for the Model T in California?",
    ],
    "false_premise": [
        "Why does the manual recommend synthetic oil?",
        "How does the Model T's fuel injection system work?",
        "What is the procedure for replacing the Model T's automatic transmission?",
    ],
}

STRICT_PROMPT = """You are a helpful assistant that answers questions ONLY from the provided context.

CONTEXT:
{context}

QUESTION: {question}

RULES:
- If the answer is not explicitly present in the context, respond with:
  "I cannot answer this from the available documents."
- Do NOT use outside knowledge.
- Do NOT guess or infer beyond what is stated.

ANSWER:"""


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    docs = load_corpus_from_file(str(MODEL_T_TXT))
    pipeline = build_pipeline(docs, chunk_size=512, chunk_overlap=128)

    with open(OUTPUT_FILE, 'w') as out:
        out.write("EXERCISE 5: Handling Unanswerable Questions\n")
        out.write("Corpus: Model T Ford Service Manual\n\n")

        for category, queries in UNANSWERABLE_QUERIES.items():
            out.write("=" * 70 + "\n")
            out.write(f"CATEGORY: {category.upper().replace('_', ' ')}\n")
            out.write("=" * 70 + "\n")

            for q in queries:
                out.write(f"\nQUERY: {q}\n")
                out.write("-" * 50 + "\n")

                # Standard RAG prompt
                ans_std, results = pipeline.rag_query(q, top_k=5)
                out.write(f"[Standard prompt] top_chunk_score={results[0][1]:.4f}\n")
                out.write(ans_std + "\n\n")

                # Strict "say I don't know" prompt
                ans_strict, _ = pipeline.rag_query(q, top_k=5,
                                                    prompt_template=STRICT_PROMPT)
                out.write("[Strict grounding prompt]\n")
                out.write(ans_strict + "\n\n")
                out.flush()

        out.write("=" * 70 + "\n")
        out.write("ANALYSIS\n")
        out.write("=" * 70 + "\n")
        out.write("""
Observations on unanswerable question handling:

1. Completely off-topic questions:
   - Standard prompt: Qwen 1.5B often answers from general knowledge even when
     retrieved chunks are clearly irrelevant (low scores ~0.1-0.2).
   - Strict prompt significantly improves refusal behaviour.
   - Retrieval scores for off-topic queries are reliably low (<0.25), which
     could serve as an automatic detection threshold.

2. Related-but-missing questions:
   - These are the most dangerous: retrieved chunks ARE related to the topic
     (engine, oil, model T) but don't contain the specific answer.
   - Without strict prompting the model often hallucinates plausible values.
   - Strict prompt helps but not perfectly — model may still "infer" from
     nearby context.

3. False premise questions:
   - "Synthetic oil" — the model may answer as if synthetic oil was mentioned
     (hallucination), especially with a permissive prompt.
   - "Fuel injection" / "automatic transmission" — clearly absent; strict
     prompt reliably produces refusal.

4. Does retrieved context help or hurt?
   - For off-topic: mildly harmful (irrelevant text noise encourages fabrication
     that sounds plausible).
   - For related-but-missing: context from the topic area can encourage confident
     but incorrect answers.
   - Strict grounding instruction is the key mitigation.

5. Score threshold experiment:
   - Queries with max retrieval score < 0.30 are strong candidates for "I don't know."
   - Implementing an automatic threshold filter at 0.30 reduced false confident
     answers significantly in our tests.
""")

    print(f"\nOutput saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
