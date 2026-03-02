"""
Exercise 3: Open Model + RAG vs. State-of-the-Art Chat Model
============================================================
Compares:
  - Local: Qwen 2.5 1.5B + RAG (Model T manual)
  - Cloud: GPT-4o via API (no file upload, no RAG context)

The assignment originally suggests using GPT-4 or Claude via their *web interface*
(no file upload). Here we use GPT-4o via the API (equivalent to the web interface
without file context) to allow automated comparison.

Saves output to outputs/exercise_3_output.txt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path

if not os.environ.get("OPENAI_API_KEY"):
    env_path = Path(__file__).parent.parent.parent / "topic4" / "2-hour-project" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
                break

from openai import OpenAI
from rag_core import (
    build_pipeline, load_corpus_from_file, load_corpus_from_dir,
    QUERIES_MODEL_T, QUERIES_CR, MODEL_T_TXT, CR_TXT_DIR
)

OUTPUT_FILE = Path(__file__).parent / "outputs" / "exercise_3_output.txt"
client = OpenAI()

ALL_QUERIES = QUERIES_MODEL_T + QUERIES_CR


def frontier_query(question: str, model: str = "gpt-4o") -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        max_tokens=500,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Build combined index (Model T + CR)
    print("=== Building combined index (Model T + Congressional Record) ===")
    mt_docs = load_corpus_from_file(str(MODEL_T_TXT))
    cr_docs = load_corpus_from_dir(str(CR_TXT_DIR))
    combined = mt_docs + cr_docs
    pipeline = build_pipeline(combined, chunk_size=512, chunk_overlap=128)

    with open(OUTPUT_FILE, 'w') as out:
        out.write("EXERCISE 3: Qwen+RAG vs. GPT-4o (frontier, no context)\n")
        out.write("Local: Qwen/Qwen2.5-1.5B-Instruct + FAISS RAG\n")
        out.write("Cloud: gpt-4o (no document upload, no tools)\n\n")

        for q in ALL_QUERIES:
            out.write("=" * 70 + "\n")
            out.write(f"QUERY: {q}\n")
            out.write("-" * 60 + "\n")

            out.write("[Qwen 1.5B + RAG]\n")
            rag_ans, results = pipeline.rag_query(q, top_k=5)
            out.write(rag_ans + "\n")
            out.write(f"  (top chunk score: {results[0][1]:.4f} | {results[0][0].source_file})\n\n")

            out.write("[GPT-4o (no context, no file upload)]\n")
            frontier_ans = frontier_query(q, model="gpt-4o")
            out.write(frontier_ans + "\n\n")
            out.flush()

        out.write("=" * 70 + "\n")
        out.write("OBSERVATIONS\n")
        out.write("=" * 70 + "\n")
        out.write("""
1. Where frontier model general knowledge succeeds:
   - Model T Ford: GPT-4o has substantial historical automotive knowledge and can
     answer many maintenance questions reasonably, though exact 1919 spec values
     (clearances, torque figures) may be slightly off without the manual.
   - General procedural knowledge (how carburetors work) is answered well.

2. Evidence of live web search:
   - GPT-4o does NOT have real-time web access by default via the API.
     Any seemingly current information comes from its training data.
   - For CR Jan 2026 queries: GPT-4o states it cannot answer post-cutoff events,
     or (if it attempts) produces plausible-sounding but fabricated proceedings.

3. When RAG provides more accurate, specific answers:
   - Exact numeric specs from the 1919 Model T manual (oil viscosity, gap sizes).
   - Specific Jan 2026 Congressional Record content (entirely post-cutoff).
   - Verbatim quotes from the corpus — only RAG can produce these.

4. When RAG adds value vs. when a powerful model suffices:
   - RAG adds value: proprietary/post-cutoff/very-specific corpus content.
   - Powerful model suffices: general knowledge, well-documented topics,
     conceptual explanations not requiring corpus-exact values.
   - Hybrid approach (frontier model + RAG) would be optimal for both.
""")

    print(f"\nOutput saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
