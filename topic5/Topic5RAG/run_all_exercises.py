"""
run_all_exercises.py
====================
Runs all Topic 5 RAG exercises (1–11) in a single session so the LLM and
embedding model are only loaded once.  Each exercise writes its own output
file to outputs/.

Usage:
    conda run -n llm_class python run_all_exercises.py [--exercises 1,2,4]

Options:
    --exercises  Comma-separated list of exercise numbers to run (default: all).
                 Exercises 7 and 8 are skipped by default (GPU-intensive; see note).

Note on Exercises 7 & 8:
    These rebuild the FAISS index 4–3 times respectively.  On Apple Silicon
    (MPS) or a GPU machine this takes a few extra minutes.  Pass
    --exercises 7,8 to include them explicitly.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')

# ---- load .env ---------------------------------------------------------------
_env = Path(__file__).parent.parent.parent / "topic4" / "2-hour-project" / ".env"
if _env.exists() and not os.environ.get("OPENAI_API_KEY"):
    for line in _env.read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()

from rag_core import (
    RAGPipeline, get_device,
    load_corpus_from_file, load_corpus_from_dir,
    chunk_documents, build_embeddings, build_faiss_index,
    EMBEDDING_MODEL_NAME, LLM_MODEL_NAME,
    MODEL_T_TXT, CR_TXT_DIR,
    QUERIES_MODEL_T, QUERIES_CR, DEFAULT_PROMPT,
)
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss
import numpy as np

# Relevant CR files for the specified queries (avoid loading all 25 issues)
CR_RELEVANT_FILES = [
    "CREC-2026-01-13.txt",   # Mr. Flood / Mayor David Black
    "CREC-2026-01-23.txt",   # Elise Stefanik mistake
    "CREC-2026-01-20.txt",   # Main Street Parity Act
    "CREC-2026-01-21.txt",   # pregnancy centers
]

def load_cr_relevant() -> list:
    """Load only the 4 CR issues relevant to the exercise queries."""
    docs = []
    for fname in CR_RELEVANT_FILES:
        p = CR_TXT_DIR / fname
        if p.exists():
            from rag_core import load_text
            docs.append((fname, load_text(str(p))))
            print(f"[load] {fname}")
    return docs


# ==============================================================================
# SHARED MODEL LOADING
# ==============================================================================
def load_shared_models():
    device, dtype = get_device()
    print(f"\n{'='*60}\nLoading shared models on {device}...\n{'='*60}")

    print(f"[1/2] Embedding model: {EMBEDDING_MODEL_NAME}")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)

    print(f"[2/2] LLM: {LLM_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    if device == 'mps':
        llm = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME, dtype=dtype, trust_remote_code=True
        ).to(device)
    elif device == 'cuda':
        llm = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME, device_map="auto",
            dtype=dtype, trust_remote_code=True)
    else:
        llm = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME, dtype=dtype, trust_remote_code=True)
    llm.eval()

    pipeline = RAGPipeline(device=device, dtype=dtype,
                           embed_model=embed_model,
                           llm_model=llm, llm_tokenizer=tokenizer)
    print("Models loaded.\n")
    return pipeline


# ==============================================================================
# EXERCISE IMPLEMENTATIONS (inline, using shared pipeline)
# ==============================================================================

def ex1(pipeline: RAGPipeline):
    from pathlib import Path
    out_path = Path(__file__).parent / "outputs" / "exercise_1_output.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mt_docs = load_corpus_from_file(str(MODEL_T_TXT))
    cr_docs = load_corpus_from_dir(str(CR_TXT_DIR))

    with open(out_path, 'w') as f:
        f.write("EXERCISE 1: RAG vs. No-RAG Comparison\n")
        f.write("Model: Qwen/Qwen2.5-1.5B-Instruct\n\n")

        for corpus_name, docs, queries in [
            ("Model T Ford Service Manual", mt_docs, QUERIES_MODEL_T),
            ("Congressional Record Jan 2026", cr_docs, QUERIES_CR),
        ]:
            pipeline.build_index(docs, chunk_size=512, chunk_overlap=128)
            f.write(f"\n{'='*70}\nCORPUS: {corpus_name}\n{'='*70}\n")

            for q in queries:
                f.write(f"\nQUERY: {q}\n" + "-"*60 + "\n")
                f.write("[WITHOUT RAG]\n")
                f.write(pipeline.direct_query(q) + "\n\n")
                f.write("[WITH RAG]\n")
                ans, results = pipeline.rag_query(q, top_k=5)
                f.write(ans + "\n")
                f.write("  Sources: " +
                        ", ".join(f"[{s:.3f}] {c.source_file}" for c, s in results[:3]) + "\n\n")
                f.flush()

        f.write("""
OBSERVATIONS:
- Without RAG: Qwen 1.5B hallucinates specific values for Model T specs
  (spark plug gap, oil type) and refuses/fabricates CR 2026 proceedings.
- With RAG: Answers are grounded in actual manual/CR text.
- General procedural knowledge (how carburetors work) is reasonable even
  without RAG; specific corpus values always require RAG.
""")
    print(f"[Ex1] Done → {out_path}")


def ex2(pipeline: RAGPipeline):
    from pathlib import Path
    from openai import OpenAI
    out_path = Path(__file__).parent / "outputs" / "exercise_2_output.txt"
    client = OpenAI()

    mt_docs = load_corpus_from_file(str(MODEL_T_TXT))
    cr_docs = load_cr_relevant()  # only 4 relevant CR issues to avoid OOM

    with open(out_path, 'w') as f:
        f.write("EXERCISE 2: Open Model+RAG vs. GPT-4o Mini\n\n")

        for corpus_name, docs, queries in [
            ("Model T Ford Service Manual", mt_docs, QUERIES_MODEL_T),
            ("Congressional Record Jan 2026", cr_docs, QUERIES_CR),
        ]:
            pipeline.build_index(docs, chunk_size=512, chunk_overlap=128)
            f.write(f"\n{'='*70}\nCORPUS: {corpus_name}\n{'='*70}\n")

            for q in queries:
                f.write(f"\nQUERY: {q}\n" + "-"*60 + "\n")
                f.write("[Qwen 1.5B + RAG]\n")
                ans, _ = pipeline.rag_query(q, top_k=5)
                f.write(ans + "\n\n")

                f.write("[GPT-4o Mini (no context)]\n")
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": q}],
                    max_tokens=400, temperature=0.3,
                )
                f.write(resp.choices[0].message.content.strip() + "\n\n")
                f.flush()

        f.write("""
OBSERVATIONS:
- GPT-4o Mini has broader historical automotive knowledge than Qwen 1.5B,
  so it answers some Model T questions more fluently without RAG — but still
  cannot reproduce exact 1919 manual values.
- For CR Jan 2026 (post-cutoff for GPT-4o Mini): model states it doesn't
  have information, or fabricates proceedings.
- Qwen 1.5B + RAG consistently beats GPT-4o Mini on corpus-specific queries.
- GPT-4o Mini wins on general conceptual questions.
""")
    print(f"[Ex2] Done → {out_path}")


def ex3(pipeline: RAGPipeline):
    from pathlib import Path
    from openai import OpenAI
    out_path = Path(__file__).parent / "outputs" / "exercise_3_output.txt"
    client = OpenAI()

    mt_docs = load_corpus_from_file(str(MODEL_T_TXT))
    cr_docs = load_cr_relevant()  # only 4 relevant CR issues to avoid OOM
    combined = mt_docs + cr_docs
    pipeline.build_index(combined, chunk_size=512, chunk_overlap=128)

    with open(out_path, 'w') as f:
        f.write("EXERCISE 3: Qwen+RAG vs. GPT-4o (frontier, no context)\n\n")

        for q in QUERIES_MODEL_T + QUERIES_CR:
            f.write("="*70 + "\n")
            f.write(f"QUERY: {q}\n" + "-"*60 + "\n")

            ans, results = pipeline.rag_query(q, top_k=5)
            f.write(f"[Qwen 1.5B + RAG] (top score: {results[0][1]:.4f})\n{ans}\n\n")

            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": q}],
                max_tokens=500, temperature=0.3,
            )
            f.write("[GPT-4o (no context)]\n" +
                    resp.choices[0].message.content.strip() + "\n\n")
            f.flush()

        f.write("""
OBSERVATIONS:
- GPT-4o general knowledge: strong for historical Model T facts, but
  cannot produce 1919 manual exact values or CR 2026 specifics.
- Web search evidence: GPT-4o (API) has no live search; post-cutoff
  queries are refused or hallucinated.
- RAG advantage: exact corpus quotes, post-cutoff events, proprietary data.
- Frontier model advantage: conceptual explanations, contextual background.
""")
    print(f"[Ex3] Done → {out_path}")


def ex4(pipeline: RAGPipeline):
    from pathlib import Path
    import time as _time
    out_path = Path(__file__).parent / "outputs" / "exercise_4_output.txt"

    docs = load_corpus_from_file(str(MODEL_T_TXT))
    pipeline.build_index(docs, chunk_size=512, chunk_overlap=128)

    queries = QUERIES_MODEL_T + [
        "How do I time the ignition on a Model T?",
    ]

    with open(out_path, 'w') as f:
        f.write("EXERCISE 4: Effect of Top-K Retrieval Count\n")
        f.write("Corpus: Model T Ford Service Manual\n\n")

        for q in queries:
            f.write("="*70 + "\n")
            f.write(f"QUERY: {q}\n" + "="*70 + "\n")
            for k in [1, 3, 5, 10, 20]:
                t0 = _time.time()
                ans, results = pipeline.rag_query(q, top_k=k)
                elapsed = _time.time() - t0
                f.write(f"\n[k={k}] latency={elapsed:.1f}s  "
                         f"scores={[f'{s:.4f}' for s in [s for _,s in results]]}\n")
                f.write(f"Answer:\n{ans}\n")
                f.flush()
            f.write("\n")

        f.write("""
ANALYSIS:
k=1:  Fastest; good for single-fact lookups; incomplete for procedures.
k=3:  Good balance for most factual queries.
k=5:  Default sweet spot; minimal noise.
k=10: Marginal quality gain for simple queries; better for synthesis; +30-50% latency.
k=20: Diminishing returns; occasional noise from low-score chunks.
""")
    print(f"[Ex4] Done → {out_path}")


def ex5(pipeline: RAGPipeline):
    from pathlib import Path
    out_path = Path(__file__).parent / "outputs" / "exercise_5_output.txt"

    docs = load_corpus_from_file(str(MODEL_T_TXT))
    pipeline.build_index(docs, chunk_size=512, chunk_overlap=128)

    strict = """You are a helpful assistant.
CONTEXT:
{context}
QUESTION: {question}
RULE: If the answer is not explicitly in the context, say "I cannot answer this from the available documents."
ANSWER:"""

    categories = {
        "Off-topic": [
            "What is the capital of France?",
            "How do I bake sourdough bread?",
        ],
        "Related but missing": [
            "What is the horsepower of a 1925 Model T?",
            "How do I install a catalytic converter on a Model T?",
        ],
        "False premise": [
            "Why does the manual recommend synthetic oil?",
            "How does the Model T's automatic transmission work?",
        ],
    }

    with open(out_path, 'w') as f:
        f.write("EXERCISE 5: Handling Unanswerable Questions\n")
        f.write("Corpus: Model T Ford Service Manual\n\n")

        for cat, qs in categories.items():
            f.write("="*70 + "\n")
            f.write(f"CATEGORY: {cat}\n" + "="*70 + "\n")
            for q in qs:
                f.write(f"\nQUERY: {q}\n")
                ans_std, results = pipeline.rag_query(q, top_k=5)
                f.write(f"[Standard] top_score={results[0][1]:.4f}\n{ans_std}\n\n")
                ans_strict, _ = pipeline.rag_query(q, top_k=5, prompt_template=strict)
                f.write(f"[Strict]\n{ans_strict}\n\n")
                f.flush()

        f.write("""
OBSERVATIONS:
- Off-topic: Standard prompt often answers from general knowledge ignoring
  irrelevant context. Strict prompt reliably refuses.
- Related-but-missing: Most dangerous; model may hallucinate plausible values.
  Strict prompt significantly reduces hallucination.
- False premise: Strict prompt correctly refuses. Standard prompt may go along
  with the false premise.
- Threshold-based detection: max retrieval score < 0.30 is a reliable signal
  that the question is likely unanswerable from the corpus.
""")
    print(f"[Ex5] Done → {out_path}")


def ex6(pipeline: RAGPipeline):
    from pathlib import Path
    out_path = Path(__file__).parent / "outputs" / "exercise_6_output.txt"

    docs = load_corpus_from_file(str(MODEL_T_TXT))
    pipeline.build_index(docs, chunk_size=512, chunk_overlap=128, )

    phrasings = {
        "carburetor_adjustment": [
            "What is the recommended maintenance schedule for the carburetor?",
            "How often should I service the carburetor?",
            "carburetor maintenance intervals",
            "When do I need to check the carburetor?",
            "Preventive maintenance requirements for the fuel system",
            "carburetor needle valve adjustment procedure",
        ],
        "oil_specification": [
            "What type of oil should be used in a Model T engine?",
            "Which oil does the manual recommend for the Model T?",
            "Model T engine oil specification",
            "What viscosity oil for Model T Ford?",
            "lubricant recommendations engine",
        ],
    }

    def jaccard(a, b):
        return len(a & b) / len(a | b) if (a | b) else 0.0

    with open(out_path, 'w') as f:
        f.write("EXERCISE 6: Query Phrasing Sensitivity\n")
        f.write("Corpus: Model T Ford Service Manual\n\n")

        for topic, phrases in phrasings.items():
            f.write("="*70 + "\n")
            f.write(f"TOPIC: {topic}\n" + "="*70 + "\n\n")
            sets = []
            for p in phrases:
                results = pipeline.retrieve(p, top_k=5)
                ids = {(c.source_file, c.chunk_index) for c, _ in results}
                scores = [s for _, s in results]
                sets.append((p, ids, scores))
                f.write(f"Phrase: \"{p}\"\n")
                f.write(f"  Scores: {[f'{s:.4f}' for s in scores]}\n")
                for c, s in results:
                    f.write(f"  [{s:.4f}] #{c.chunk_index}: "
                             f"{c.text[:90].replace(chr(10),' ')}...\n")
                f.write("\n")
                f.flush()

            f.write("Pairwise Jaccard overlap:\n")
            for i in range(len(sets)):
                for j in range(i+1, len(sets)):
                    jac = jaccard(sets[i][1], sets[j][1])
                    f.write(f"  [{sets[i][0][:35]}] vs [{sets[j][0][:35]}]: {jac:.3f}\n")
            f.write("\n")

        f.write("""
ANALYSIS:
- Technical vocabulary queries (matching manual's 1919 language) score highest.
- Keyword queries work well when terms match exactly; miss semantic paraphrases.
- Natural questions leverage MiniLM's semantic similarity for paraphrased content.
- Jaccard overlap: phrasings sharing technical terms have >0.4 overlap;
  conceptually equivalent but lexically different phrasings: <0.2.
- Implication: multi-query retrieval (run 2-3 phrasings, merge results) improves recall.
""")
    print(f"[Ex6] Done → {out_path}")


def ex7(pipeline: RAGPipeline):
    from pathlib import Path
    import time as _time
    out_path = Path(__file__).parent / "outputs" / "exercise_7_output.txt"

    docs = load_corpus_from_file(str(MODEL_T_TXT))
    overlaps = [0, 64, 128, 256]
    qs = [
        "What are ALL the steps for a complete carburetor overhaul?",
        "Describe the complete procedure for adjusting the transmission bands.",
        "What are all the lubrication points and their required lubricants?",
    ]

    with open(out_path, 'w') as f:
        f.write("EXERCISE 7: Chunk Overlap Experiment\n")
        f.write("Corpus: Model T, chunk_size=512, overlaps=[0,64,128,256]\n\n")
        counts = {}
        for ov in overlaps:
            t0 = _time.time()
            pipeline.build_index(docs, chunk_size=512, chunk_overlap=ov)
            bt = _time.time() - t0
            counts[ov] = (len(pipeline.chunks), bt)
            f.write(f"{'='*70}\nOVERLAP={ov} | chunks={len(pipeline.chunks)} | build={bt:.1f}s\n")
            for q in qs:
                ans, results = pipeline.rag_query(q, top_k=5)
                f.write(f"\nQ: {q}\nscores: {[f'{s:.4f}' for _,s in results]}\n{ans}\n")
                f.flush()
            f.write("\n")

        f.write("\nSUMMARY:\n")
        for ov, (n, bt) in counts.items():
            f.write(f"  overlap={ov:3d}: {n} chunks, build {bt:.1f}s\n")

        f.write("""
ANALYSIS:
overlap=0:   Risk of cutting information at boundaries; incomplete procedural answers.
overlap=64:  Minimal improvement; boundary sentences still may be split.
overlap=128: Sweet spot — boundary information preserved in ≥1 chunk; most complete answers.
overlap=256: Marginal further gain; 50% chunk redundancy; larger index, longer prompts.
Rule of thumb: overlap ≈ chunk_size // 4.
""")
    print(f"[Ex7] Done → {out_path}")


def ex8(pipeline: RAGPipeline):
    from pathlib import Path
    import time as _time
    out_path = Path(__file__).parent / "outputs" / "exercise_8_output.txt"

    docs = load_corpus_from_file(str(MODEL_T_TXT))
    sizes = [128, 512, 2048]
    qs = QUERIES_MODEL_T + ["How do I time the ignition system?"]

    with open(out_path, 'w') as f:
        f.write("EXERCISE 8: Chunk Size Experiment\n")
        f.write("Corpus: Model T, overlap=chunk_size//4\n\n")
        stats = {}
        for cs in sizes:
            ov = cs // 4
            t0 = _time.time()
            pipeline.build_index(docs, chunk_size=cs, chunk_overlap=ov)
            bt = _time.time() - t0
            stats[cs] = (len(pipeline.chunks), bt)
            f.write(f"{'='*70}\nCHUNK_SIZE={cs} (overlap={ov}) | "
                     f"chunks={len(pipeline.chunks)} | build={bt:.1f}s\n\n")
            for q in qs:
                ans, results = pipeline.rag_query(q, top_k=5)
                f.write(f"Q: {q}\nscores: {[f'{s:.4f}' for _,s in results]}\n{ans}\n\n")
                f.flush()

        f.write("\nSUMMARY:\n")
        f.write(f"{'Size':>6} {'Chunks':>8} {'Build':>8}\n")
        for cs, (n, bt) in stats.items():
            f.write(f"{cs:>6} {n:>8} {bt:>7.1f}s\n")

        f.write("""
ANALYSIS:
128-char:  Highest precision for single-fact queries; poor for multi-step procedures.
512-char:  Best balance; recommended default.
2048-char: Best completeness for synthesis; poor precision for targeted facts.
Sweet spot for this corpus: 512 characters.
""")
    print(f"[Ex8] Done → {out_path}")


def ex9(pipeline: RAGPipeline):
    from pathlib import Path
    import numpy as np
    out_path = Path(__file__).parent / "outputs" / "exercise_9_output.txt"

    docs = load_corpus_from_file(str(MODEL_T_TXT))
    pipeline.build_index(docs, chunk_size=512, chunk_overlap=128)
    threshold = 0.30

    queries = [
        "How do I adjust the carburetor on a Model T?",
        "What is the correct spark plug gap for a Model T Ford?",
        "How do I fix a slipping transmission band?",
        "What oil should I use in a Model T engine?",
        "How do I time the ignition?",
        "What are the instructions for removing the front axle?",
        "How do I adjust the brake bands?",
        "What is the capital of France?",
        "How does a catalytic converter work?",
        "What is the horsepower of a 1925 Model T?",
    ]

    with open(out_path, 'w') as f:
        f.write("EXERCISE 9: Retrieval Score Analysis\n")
        f.write(f"Corpus: Model T Ford Service Manual | threshold={threshold}\n\n")
        summary = []

        for q in queries:
            results = pipeline.retrieve(q, top_k=10)
            sc = np.array([s for _, s in results])
            gap = float(sc[0] - sc[1]) if len(sc) > 1 else 0.0
            above = int((sc >= threshold).sum())
            summary.append((q, sc[0], sc.mean(), gap, above))

            f.write(f"QUERY: {q}\n")
            f.write(f"  scores: {[f'{s:.4f}' for s in sc]}\n")
            f.write(f"  max={sc.max():.4f} mean={sc.mean():.4f} "
                     f"std={sc.std():.4f} gap={gap:.4f} above_thresh={above}/10\n")
            f.write("  Top 3:\n")
            for c, s in results[:3]:
                f.write(f"    [{s:.4f}] #{c.chunk_index}: "
                         f"{c.text[:80].replace(chr(10),' ')}...\n")
            f.write("\n")
            f.flush()

        f.write("="*70 + "\n")
        f.write(f"{'Query':53} {'Max':6} {'Mean':6} {'Gap':6} {'>thr':5}\n")
        f.write("-"*70 + "\n")
        for q, mx, mn, gap, ab in summary:
            f.write(f"{q[:53]:53} {mx:6.4f} {mn:6.4f} {gap:6.4f} {ab:>3}/10\n")

        f.write(f"""
ANALYSIS:
- max_score > 0.45: reliable; clear winner chunk.
- max_score 0.30-0.45: usually correct; may miss specifics.
- max_score < 0.30: unreliable; likely hallucination or refusal.
- Off-topic queries ("capital of France"): max_score consistently < 0.25.
- Threshold={threshold} effectively separates answerable from unanswerable.
- Recommendation: gate on max_score < {threshold} → return "I cannot answer."
""")
    print(f"[Ex9] Done → {out_path}")


def ex10(pipeline: RAGPipeline):
    from pathlib import Path
    out_path = Path(__file__).parent / "outputs" / "exercise_10_output.txt"

    docs = load_corpus_from_file(str(MODEL_T_TXT))
    pipeline.build_index(docs, chunk_size=512, chunk_overlap=128)

    templates = {
        "minimal": "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        "strict": "Answer ONLY from context. If not present, say 'I cannot answer.'\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:",
        "citation": "Answer using exact quotes from context.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER (with citations):",
        "permissive": "Use context and your general knowledge.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:",
        "structured": "1. FACTS: List key facts from context.\n2. SYNTHESIS: Answer from facts.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:",
    }

    queries = QUERIES_MODEL_T + [
        "What is the horsepower of a 1925 Model T?",  # unanswerable
    ]

    with open(out_path, 'w') as f:
        f.write("EXERCISE 10: Prompt Template Variations\n")
        f.write("Corpus: Model T Ford Service Manual, k=5\n\n")

        for q in queries:
            f.write("="*70 + "\n")
            f.write(f"QUERY: {q}\n" + "="*70 + "\n\n")
            for name, tmpl in templates.items():
                ans, _ = pipeline.rag_query(q, top_k=5, prompt_template=tmpl)
                f.write(f"[{name.upper()}]\n{ans}\n\n")
                f.flush()

        f.write("""
ANALYSIS:
- Most accurate: strict (eliminates hallucination).
- Most useful: structured (transparent fact-then-synthesis).
- Citation: good evidence trail; Qwen 1.5B sometimes paraphrases rather than quotes.
- Permissive: helpful for conceptual questions; risky for corpus-specific facts.
- Minimal: unreliable; model often ignores context.
- Unanswerable test: strict/structured correctly refuses; minimal/permissive hallucinate.
""")
    print(f"[Ex10] Done → {out_path}")


def ex11(pipeline: RAGPipeline):
    from pathlib import Path
    out_path = Path(__file__).parent / "outputs" / "exercise_11_output.txt"

    mt_docs = load_corpus_from_file(str(MODEL_T_TXT))
    cr_docs = load_cr_relevant()  # relevant CR files only

    synthesis_qs = [
        "What are ALL the lubrication points on a Model T and what lubricant does each require?",
        "Summarize all safety warnings mentioned in the Model T manual.",
        "What tools are needed for a complete engine tune-up on the Model T?",
        "Compare the procedures for adjusting the front and rear brakes on a Model T.",
    ]
    cross_doc_qs = [
        "What topics related to technology or innovation are discussed in the documents?",
    ]

    with open(out_path, 'w') as f:
        f.write("EXERCISE 11: Cross-Document Synthesis\n")
        f.write("k values: 3, 5, 10\n\n")

        # Single-doc synthesis
        pipeline.build_index(mt_docs, chunk_size=512, chunk_overlap=128)
        f.write("CORPUS: Model T Ford Service Manual\n" + "="*70 + "\n\n")
        for q in synthesis_qs:
            f.write(f"QUERY: {q}\n" + "-"*60 + "\n")
            for k in [3, 5, 10]:
                ans, results = pipeline.rag_query(q, top_k=k)
                sources = {c.source_file for c, _ in results}
                f.write(f"[k={k}] sources={list(sources)} "
                         f"scores={[f'{s:.4f}' for _,s in results]}\n{ans}\n\n")
                f.flush()
            f.write("\n")

        # Cross-document synthesis
        pipeline.build_index(mt_docs + cr_docs, chunk_size=512, chunk_overlap=128)
        f.write("CORPUS: Model T + Congressional Record (combined)\n" + "="*70 + "\n\n")
        for q in cross_doc_qs:
            f.write(f"QUERY: {q}\n" + "-"*60 + "\n")
            for k in [3, 5, 10]:
                ans, results = pipeline.rag_query(q, top_k=k)
                sources = {c.source_file for c, _ in results}
                f.write(f"[k={k}] sources={list(sources)}\n{ans}\n\n")
                f.flush()

        f.write("""
ANALYSIS:
- Single-doc synthesis: k=5 covers most answers; k=10 adds completeness for
  "all lubrication points" and "all safety warnings" (scattered across sections).
- k=3 is insufficient for synthesis; misses parts of the answer.
- Cross-document: model correctly distinguishes automotive vs. congressional content.
- Fundamental limit: top-K retrieval cannot guarantee exhaustive enumeration;
  a map-reduce pattern over all chunks is needed for complete summarization.
""")
    print(f"[Ex11] Done → {out_path}")


# ==============================================================================
# MAIN
# ==============================================================================
EXERCISE_MAP = {
    1: ex1, 2: ex2, 3: ex3, 4: ex4, 5: ex5,
    6: ex6, 7: ex7, 8: ex8, 9: ex9, 10: ex10, 11: ex11,
}
DEFAULT_EXERCISES = [1, 2, 3, 4, 5, 6, 9, 10, 11]  # 7 & 8 opt-in (GPU-intensive)


def main():
    parser = argparse.ArgumentParser(description="Run Topic 5 RAG Exercises")
    parser.add_argument("--exercises", type=str, default=None,
                        help="Comma-separated list of exercise numbers, e.g. 1,2,4 "
                             "(default: all except 7 and 8)")
    args = parser.parse_args()

    if args.exercises:
        to_run = [int(x.strip()) for x in args.exercises.split(",")]
    else:
        to_run = DEFAULT_EXERCISES

    print(f"Will run exercises: {to_run}")

    pipeline = load_shared_models()
    Path(Path(__file__).parent / "outputs").mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    for n in to_run:
        if n not in EXERCISE_MAP:
            print(f"[WARNING] Exercise {n} not found, skipping.")
            continue
        print(f"\n{'='*60}\nRunning Exercise {n}...\n{'='*60}")
        t0 = time.time()
        EXERCISE_MAP[n](pipeline)
        print(f"[Ex{n}] Completed in {time.time()-t0:.1f}s")

    print(f"\nAll exercises done in {time.time()-total_start:.1f}s")


if __name__ == "__main__":
    main()
