# Task 1 — Running an LLM

Benchmarking and chatting with small language models locally on a MacBook (Apple Silicon).

## Files

| File | Description |
|---|---|
| `llama_mmlu_eval.py` | Multi-model MMLU evaluation script (3 models × 10 subjects, timing, `--verbose` flag) |
| `barebone_chatAgent.py` | Chat agent with hybrid context management and `USE_CONTEXT_HISTORY` toggle |
| `analyze_results.py` | Generates PNG accuracy/difficulty graphs from JSON results |
| `results_analysis/` | PNG graphs (accuracy by subject, overall accuracy, subject difficulty) |
| `multi_model_mmlu_results_*.json` | Full evaluation results (per-question details + timing) |
| `llama_3.2_1b_mmlu_results_*.json` | Early single-model runs (MPS full-precision and CPU 4-bit) |

## Portfolio

See [`Running an LLM/README.md`](Running%20an%20LLM/README.md) for the full write-up including:
- Setup instructions and environment details
- Timing comparison (GPU MPS vs CPU 4-bit)
- Model accuracy tables and per-subject breakdown
- Mistake pattern analysis
- Chat agent context management comparison
- PDF graphs in `Running an LLM/graphs/`

## Quick Start

```bash
conda activate llm-class
huggingface-cli login

# Run evaluation (3 models, 10 subjects)
python llama_mmlu_eval.py

# Run with verbose question-by-question output
python llama_mmlu_eval.py --verbose

# Run chat agent
python barebone_chatAgent.py

# Generate PDF graphs
python "Running an LLM/generate_pdf_graphs.py"
```

## Results Summary

| Model | Accuracy | Real Time (MPS) |
|---|---|---|
| Qwen2.5-0.5B | **43.4%** | 53.9s |
| Llama-3.2-1B-Instruct | 43.3% | 76.4s |
| google/gemma-2b | 34.1% | 113.3s |

*1,336 questions across 10 MMLU subjects — Apple MPS, FP16*
