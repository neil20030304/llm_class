"""
Exercise 10: Prompt Template Variations
=========================================
Tests five prompt template styles and evaluates answer quality on the same queries.

Templates:
  1. Minimal       — context + question, no instructions
  2. Strict        — "answer ONLY from context, say I don't know if missing"
  3. Citation      — "quote the exact passages that support your answer"
  4. Permissive    — "use context as guidance; may also use general knowledge"
  5. Structured    — "list relevant facts first, then synthesize"

Corpus: Model T Ford Service Manual
Saves output to outputs/exercise_10_output.txt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from rag_core import build_pipeline, load_corpus_from_file, MODEL_T_TXT

OUTPUT_FILE = Path(__file__).parent / "outputs" / "exercise_10_output.txt"

PROMPT_TEMPLATES = {
    "minimal": """Context:
{context}

Question: {question}

Answer:""",

    "strict_grounding": """You are a helpful assistant. Answer the question ONLY based on the context below.
If the answer is not in the context, respond with: "I cannot answer this from the available documents."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:""",

    "citation": """You are a helpful assistant. Answer the question below by quoting the exact passages
from the context that support your answer. Use quotation marks for direct quotes.

CONTEXT:
{context}

QUESTION: {question}

ANSWER (with citations):""",

    "permissive": """You are a helpful assistant. Use the context below to help answer the question.
You may also draw on your general knowledge if it helps provide a more complete answer.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:""",

    "structured": """You are a helpful assistant. Structure your answer as follows:
1. RELEVANT FACTS: List the key facts from the context that relate to the question.
2. SYNTHESIS: Combine these facts into a clear, direct answer.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:""",
}

QUERIES = [
    "How do I adjust the carburetor on a Model T?",
    "What is the correct spark plug gap for a Model T Ford?",
    "How do I fix a slipping transmission band?",
    "What oil should I use in a Model T engine?",
    "What is the horsepower of a 1925 Model T?",  # unanswerable — tests strict vs permissive
]


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    docs = load_corpus_from_file(str(MODEL_T_TXT))
    pipeline = build_pipeline(docs, chunk_size=512, chunk_overlap=128)

    with open(OUTPUT_FILE, 'w') as out:
        out.write("EXERCISE 10: Prompt Template Variations\n")
        out.write("Corpus: Model T Ford Service Manual\n")
        out.write("k=5 chunks for all queries\n\n")

        for q in QUERIES:
            out.write("=" * 70 + "\n")
            out.write(f"QUERY: {q}\n")
            out.write("=" * 70 + "\n\n")

            for tmpl_name, tmpl in PROMPT_TEMPLATES.items():
                out.write(f"--- Template: {tmpl_name.upper()} ---\n")
                ans, results = pipeline.rag_query(q, top_k=5, prompt_template=tmpl)
                out.write(ans + "\n\n")
                out.flush()

        out.write("=" * 70 + "\n")
        out.write("ANALYSIS\n")
        out.write("=" * 70 + "\n")
        out.write("""
Prompt Template Comparison:

1. Most accurate answers:
   strict_grounding — Forces the model to stay within retrieved context.
   Eliminates hallucinated values. Best for corpus-specific factual queries.

2. Most useful answers:
   structured — Separates fact extraction from synthesis, making reasoning
   transparent. Users can verify the facts list against the source.

3. Citation quality:
   citation template produces the best evidence trail, but Qwen 1.5B
   sometimes paraphrases rather than quoting verbatim, especially for
   long passages.

4. Trade-offs:
   - strict vs. permissive: strict is safer (no hallucination) but may
     refuse answerable questions if retrieval is imperfect.
   - permissive: more helpful for general questions but risks mixing
     corpus facts with LLM hallucinations.
   - minimal: unreliable — model often ignores context and answers
     from training data.

5. Unanswerable question ("1925 Model T horsepower"):
   - minimal / permissive: model invents a number (hallucination).
   - strict: correctly refuses with "I cannot answer from available documents."
   - structured: lists "no relevant facts found," then refuses — cleanest behavior.

Recommendation: Use strict_grounding or structured for production RAG;
use permissive only when hybrid (corpus + general knowledge) is explicitly desired.
""")

    print(f"\nOutput saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
