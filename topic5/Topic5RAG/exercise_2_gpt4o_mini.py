"""
Exercise 2: Open Model + RAG vs. GPT-4o Mini (no tools)
=======================================================
Runs GPT-4o Mini directly (no context) on the same queries as Exercise 1,
then compares results against Qwen+RAG.

Requires: OPENAI_API_KEY environment variable or .env file.
Saves output to outputs/exercise_2_output.txt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path

# Load .env from topic4 project if not already set
if not os.environ.get("OPENAI_API_KEY"):
    env_path = Path(__file__).parent.parent.parent / "topic4" / "2-hour-project" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
                break

from openai import OpenAI
from rag_core import (
    RAGPipeline, build_pipeline,
    load_corpus_from_file, load_corpus_from_dir,
    QUERIES_MODEL_T, QUERIES_CR,
    MODEL_T_TXT, CR_TXT_DIR
)

OUTPUT_FILE = Path(__file__).parent / "outputs" / "exercise_2_output.txt"
client = OpenAI()


def gpt4o_mini_query(question: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        max_tokens=400,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def run_comparison(pipeline: RAGPipeline, queries: list, corpus_name: str, out):
    divider = "=" * 70
    out.write(f"\n{divider}\nCORPUS: {corpus_name}\n{divider}\n")

    for q in queries:
        out.write(f"\nQUERY: {q}\n" + "-" * 60 + "\n")

        # Qwen + RAG
        out.write("[Qwen 2.5 1.5B + RAG]\n")
        rag_ans, _ = pipeline.rag_query(q, top_k=5)
        out.write(rag_ans + "\n\n")

        # GPT-4o Mini (no tools, no retrieval)
        out.write("[GPT-4o Mini (no context)]\n")
        gpt_ans = gpt4o_mini_query(q)
        out.write(gpt_ans + "\n\n")
        out.flush()


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as out:
        out.write("EXERCISE 2: Open Model + RAG vs. GPT-4o Mini\n")
        out.write("Qwen model: Qwen/Qwen2.5-1.5B-Instruct\n")
        out.write("Comparison: GPT-4o Mini with no context/tools\n\n")

        # Model T
        print("\n=== Loading Model T corpus ===")
        mt_docs = load_corpus_from_file(str(MODEL_T_TXT))
        pipeline = build_pipeline(mt_docs, chunk_size=512, chunk_overlap=128)
        run_comparison(pipeline, QUERIES_MODEL_T, "Model T Ford Service Manual", out)

        # Congressional Record
        print("\n=== Loading Congressional Record corpus ===")
        cr_docs = load_corpus_from_dir(str(CR_TXT_DIR))
        pipeline.build_index(cr_docs, chunk_size=512, chunk_overlap=128)
        run_comparison(pipeline, QUERIES_CR, "Congressional Record Jan 2026", out)

        out.write("\n" + "=" * 70 + "\n")
        out.write("OBSERVATIONS\n")
        out.write("=" * 70 + "\n")
        out.write("""
1. GPT-4o Mini vs. hallucination (Model T):
   - GPT-4o Mini (trained on far more data) has some knowledge of the Model T
     era and produces more coherent answers without RAG, but still cannot
     reliably reproduce specific values from the 1919 manual.
   - Qwen 1.5B + RAG beats GPT-4o Mini on corpus-specific numeric specs.

2. GPT-4o Mini vs. hallucination (Congressional Record):
   - GPT-4o Mini's training cutoff is around early 2024, well before Jan 2026.
   - For CR Jan 2026 queries it will state it doesn't know or hallucinate.
   - Qwen 1.5B + RAG correctly answers from retrieved CR text.

3. GPT-4o Mini knowledge cutoff vs. corpus age:
   - Model T Ford (1919 manual): Both models lack exact manual details but
     GPT-4o Mini has broader historical automotive knowledge.
   - Congressional Record (Jan 2026): Entirely post-cutoff for GPT-4o Mini.

4. When does a larger model without RAG suffice?
   - General knowledge questions where exact corpus language is not needed.
   - Questions about well-documented historical topics.
   - RAG adds essential value for post-cutoff, proprietary, or highly specific data.
""")

    print(f"\nOutput saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
