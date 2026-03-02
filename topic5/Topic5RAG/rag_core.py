"""
rag_core.py — Shared RAG pipeline utilities for Topic 5 exercises.

Usage in exercise scripts:
    from rag_core import build_pipeline, retrieve, rag_query, direct_query, RAGPipeline
"""

import os
import pickle
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')

import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
CORPUS_ROOT = REPO_ROOT / "Corpora"
MODEL_T_TXT   = CORPUS_ROOT / "ModelTService" / "txt" / "Ford-Model-T-Man-1919.txt"
CR_TXT_DIR    = CORPUS_ROOT / "Congressional_Record_Jan_2026" / "txt"
LEARJET_TXT_DIR = CORPUS_ROOT / "Learjet" / "txt"
EU_AI_TXT     = CORPUS_ROOT / "EU_AI_Act.txt"   # fallback

# Official query lists (Exercise 1)
QUERIES_MODEL_T = [
    "How do I adjust the carburetor on a Model T?",
    "What is the correct spark plug gap for a Model T Ford?",
    "How do I fix a slipping transmission band?",
    "What oil should I use in a Model T engine?",
]
QUERIES_CR = [
    "What did Mr. Flood have to say about Mayor David Black in Congress on January 13, 2026?",
    "What mistake did Elise Stefanik make in Congress on January 23, 2026?",
    "What is the purpose of the Main Street Parity Act?",
    "Who in Congress has spoken for and against funding of pregnancy centers?",
]

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME       = "Qwen/Qwen2.5-1.5B-Instruct"

DEFAULT_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer the question based ONLY on the information in the context above
- If the context does not contain enough information to answer, say so clearly
- Quote relevant parts of the context to support your answer
- Be concise and direct

ANSWER:"""

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
def get_device() -> Tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        print(f"[device] CUDA: {torch.cuda.get_device_name(0)}")
        return 'cuda', torch.float16
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("[device] MPS (Apple Silicon)")
        return 'mps', torch.float32
    else:
        print("[device] CPU")
        return 'cpu', torch.float32


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------
def load_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def load_corpus_from_file(txt_path: str) -> List[Tuple[str, str]]:
    """Load a single .txt file as a corpus."""
    content = load_text(txt_path)
    return [(Path(txt_path).name, content)]


def load_corpus_from_dir(txt_dir: str, pattern: str = "*.txt") -> List[Tuple[str, str]]:
    """Load all .txt files from a directory."""
    docs = []
    for p in sorted(Path(txt_dir).glob(pattern)):
        docs.append((p.name, load_text(str(p))))
    print(f"[load] {len(docs)} files from {txt_dir}")
    return docs


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    text: str
    source_file: str
    chunk_index: int
    start_char: int
    end_char: int


def chunk_text(text: str, source_file: str,
               chunk_size: int = 512, chunk_overlap: int = 128) -> List[Chunk]:
    chunks: List[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            para = text.rfind('\n\n', start + chunk_size // 2, end)
            if para != -1:
                end = para + 2
            else:
                sent = text.rfind('. ', start + chunk_size // 2, end)
                if sent != -1:
                    end = sent + 2
        fragment = text[start:end].strip()
        if fragment:
            chunks.append(Chunk(fragment, source_file, idx, start, end))
            idx += 1
        new_start = end - chunk_overlap
        if chunks and new_start <= chunks[-1].start_char:
            new_start = end
        start = new_start
    return chunks


def chunk_documents(documents: List[Tuple[str, str]],
                    chunk_size: int = 512, chunk_overlap: int = 128) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for fname, content in documents:
        c = chunk_text(content, fname, chunk_size, chunk_overlap)
        all_chunks.extend(c)
    return all_chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def build_embeddings(chunks: List[Chunk], model: SentenceTransformer,
                     batch_size: int = 256) -> np.ndarray:
    texts = [c.text for c in chunks]
    parts = []
    for i in range(0, len(texts), batch_size):
        parts.append(model.encode(texts[i:i+batch_size], show_progress_bar=False))
        if (i // batch_size + 1) % 10 == 0:
            print(f"  embedded {min(i+batch_size, len(texts))}/{len(texts)}")
    return np.vstack(parts).astype('float32')


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    idx = faiss.IndexFlatIP(dim)
    idx.add(embeddings)
    return idx


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------
class RAGPipeline:
    def __init__(self, device: str = None, dtype: torch.dtype = None,
                 embed_model: SentenceTransformer = None,
                 llm_model=None, llm_tokenizer=None):
        if device is None:
            device, dtype = get_device()
        self.device = device
        self.dtype  = dtype
        self.embed_model   = embed_model
        self.llm_model     = llm_model
        self.llm_tokenizer = llm_tokenizer
        self.chunks: List[Chunk] = []
        self.index: Optional[faiss.IndexFlatIP] = None

    # ---- model loading ----
    def load_embed_model(self):
        print(f"[embed] Loading {EMBEDDING_MODEL_NAME} on {self.device}...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=self.device)
        print(f"[embed] dim={self.embed_model.get_sentence_embedding_dimension()}")

    def load_llm(self):
        print(f"[llm] Loading {LLM_MODEL_NAME} on {self.device}...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        if self.device == 'cuda':
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME, device_map="auto",
                dtype=self.dtype, trust_remote_code=True)
        elif self.device == 'mps':
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME, dtype=self.dtype, trust_remote_code=True
            ).to(self.device)
        else:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME, dtype=self.dtype, trust_remote_code=True)
        self.llm_model.eval()
        print("[llm] Ready.")

    # ---- index building ----
    def build_index(self, documents: List[Tuple[str, str]],
                    chunk_size: int = 512, chunk_overlap: int = 128):
        self.chunks = chunk_documents(documents, chunk_size, chunk_overlap)
        print(f"[index] {len(self.chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
        emb = build_embeddings(self.chunks, self.embed_model)
        self.index = build_faiss_index(emb)
        print(f"[index] FAISS index built ({self.index.ntotal} vectors)")

    def save_index(self, path: str):
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.chunks", 'wb') as f:
            pickle.dump(self.chunks, f)

    def load_index(self, path: str):
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.chunks", 'rb') as f:
            self.chunks = pickle.load(f)

    # ---- retrieval ----
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        q_emb = self.embed_model.encode([query]).astype('float32')
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, top_k)
        return [(self.chunks[i], float(s))
                for s, i in zip(scores[0], indices[0]) if i != -1]

    # ---- generation ----
    def generate(self, prompt: str, max_new_tokens: int = 400,
                 temperature: float = 0.3) -> str:
        inputs = self.llm_tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.llm_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=self.llm_tokenizer.eos_token_id,
            )
        return self.llm_tokenizer.decode(
            out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
        ).strip()

    def direct_query(self, question: str, max_new_tokens: int = 400) -> str:
        prompt = f"Answer this question:\n{question}\n\nAnswer:"
        return self.generate(prompt, max_new_tokens=max_new_tokens)

    def rag_query(self, question: str, top_k: int = 5,
                  prompt_template: str = None,
                  show_context: bool = False,
                  max_new_tokens: int = 400) -> Tuple[str, List[Tuple[Chunk, float]]]:
        results = self.retrieve(question, top_k)
        ctx = "\n\n---\n\n".join(
            f"[Source: {c.source_file}, Score: {s:.3f}]\n{c.text}"
            for c, s in results
        )
        if show_context:
            print("=== CONTEXT ===")
            print(ctx[:2000])
            print("===============")
        tmpl = prompt_template or DEFAULT_PROMPT
        prompt = tmpl.format(context=ctx, question=question)
        return self.generate(prompt, max_new_tokens=max_new_tokens), results


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------
def build_pipeline(documents: List[Tuple[str, str]],
                   chunk_size: int = 512, chunk_overlap: int = 128,
                   load_llm: bool = True) -> RAGPipeline:
    p = RAGPipeline()
    p.load_embed_model()
    if load_llm:
        p.load_llm()
    p.build_index(documents, chunk_size, chunk_overlap)
    return p
