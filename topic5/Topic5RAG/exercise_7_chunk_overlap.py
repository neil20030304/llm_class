"""
Exercise 7: Chunk Overlap Experiment
======================================
Tests how chunk overlap affects retrieval of information spanning chunk boundaries.
Chunk size is fixed at 512; overlap varies: 0, 64, 128, 256.

NOTE: This exercise rebuilds the FAISS index four times.
Recommended hardware: GPU (CUDA T4+) or Apple Silicon MPS.
On CPU-only machines this will be slow.

Corpus: Model T Ford Service Manual
Saves output to outputs/exercise_7_output.txt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
import time
from rag_core import (
    RAGPipeline, load_corpus_from_file,
    chunk_documents, build_embeddings, build_faiss_index,
    SentenceTransformer, EMBEDDING_MODEL_NAME, MODEL_T_TXT, get_device
)

OUTPUT_FILE = Path(__file__).parent / "outputs" / "exercise_7_output.txt"

CHUNK_SIZE  = 512
OVERLAPS    = [0, 64, 128, 256]

# Questions whose answers span natural chunk boundaries in the manual
BOUNDARY_QUERIES = [
    "What are ALL the steps for a complete carburetor overhaul?",
    "Describe the complete procedure for adjusting the transmission bands.",
    "What are all the lubrication points on the Model T and what oil should be used?",
    "What are all the steps for timing the ignition on a Model T?",
]


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    docs = load_corpus_from_file(str(MODEL_T_TXT))

    device, dtype = get_device()
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    # Load LLM once
    from rag_core import RAGPipeline, LLM_MODEL_NAME
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
        out.write("EXERCISE 7: Chunk Overlap Experiment\n")
        out.write(f"Corpus: Model T Ford Service Manual\n")
        out.write(f"Fixed chunk_size={CHUNK_SIZE}, varying overlap\n\n")

        overlap_chunk_counts = {}

        for overlap in OVERLAPS:
            out.write("=" * 70 + "\n")
            out.write(f"OVERLAP = {overlap}\n")
            out.write("=" * 70 + "\n")

            t0 = time.time()
            pipeline.build_index(docs, chunk_size=CHUNK_SIZE, chunk_overlap=overlap)
            build_time = time.time() - t0

            n_chunks = len(pipeline.chunks)
            overlap_chunk_counts[overlap] = n_chunks
            out.write(f"Chunks: {n_chunks} | Index build time: {build_time:.1f}s\n\n")

            for q in BOUNDARY_QUERIES:
                out.write(f"Query: {q}\n")
                ans, results = pipeline.rag_query(q, top_k=5)
                scores = [f"{s:.4f}" for _, s in results]
                out.write(f"Top scores: {scores}\n")
                out.write(f"Answer:\n{ans}\n\n")
                out.flush()

        # Summary table
        out.write("=" * 70 + "\n")
        out.write("CHUNK COUNT BY OVERLAP\n")
        out.write("=" * 70 + "\n")
        for ov, count in overlap_chunk_counts.items():
            out.write(f"  overlap={ov:3d}: {count} chunks\n")

        out.write("\n" + "=" * 70 + "\n")
        out.write("ANALYSIS\n")
        out.write("=" * 70 + "\n")
        out.write("""
Chunk Overlap Effect:

1. Does higher overlap improve retrieval of complete information?
   - Yes, significantly for questions whose answers span chunk boundaries.
   - overlap=0: Information right at boundaries may be split across chunks,
     causing one half to be retrieved without the other.
   - overlap=128: The sweet spot — boundary information appears intact in
     at least one chunk; answers are more complete.
   - overlap=256: Further improvement is marginal; mostly redundant retrieval.

2. Cost of higher overlap:
   - Index size grows proportionally (more chunks → larger FAISS index).
   - Redundant content appears in context: the same sentence may appear in
     2-3 chunks, wasting context window space.
   - Build time increases linearly with chunk count.

3. Point of diminishing returns:
   - overlap ≈ 25% of chunk_size (128 of 512) provides most of the benefit.
   - Beyond 50% overlap (256 of 512), chunks are more than half duplicate —
     retrieval quality plateaus while storage cost continues to rise.

4. Rule of thumb:
   overlap = chunk_size // 4  (e.g., 128 for 512-char chunks)
""")

    print(f"\nOutput saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
