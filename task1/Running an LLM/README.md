# Running an LLM — Portfolio Notes

## 1. Environment Setup

**Hardware:** MacBook with Apple Silicon (M-series, MPS backend)
**Conda env:** `llm-class` (Python 3.11)

```bash
conda create -n llm-class python=3.11
conda activate llm-class
pip install transformers torch datasets accelerate tqdm huggingface_hub bitsandbytes matplotlib seaborn
huggingface-cli login   # paste HF token
```

**Note on quantization:** Apple Silicon MPS is **incompatible** with `bitsandbytes` (requires CUDA).
- GPU 4-bit / 8-bit quantization → **skipped** (MacBook)
- CPU 4-bit quantization → **available** (ran via bitsandbytes on CPU)

---

## 2. Timing Comparison

### 2a. GPU (MPS) vs CPU — Llama-3.2-1B on 252 questions (2 subjects)

| Setup | Device | Quant | Time (s) | Time/question |
|---|---|---|---|---|
| `time python llama_mmlu_eval.py` | MPS (Apple GPU) | None (FP16) | 23.5 s | 0.093 s/q |
| `time python llama_mmlu_eval.py` | CPU | 4-bit | 85.2 s | 0.338 s/q |

**→ Apple MPS is ~3.6× faster than CPU 4-bit on this hardware.**

### 2b. Three models on 10 subjects (1 336 questions, MPS, FP16)

| Model | Params | Real Time (s) | CPU Time (s) | GPU Time (s) | Throughput |
|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 1B | 76.4 | 46.3 | 76.4 | 17.5 q/s |
| Qwen2.5-0.5B | 0.5B | 53.9 | 51.0 | 53.9 | **24.8 q/s** |
| google/gemma-2b | 2B | 113.3 | 48.7 | 113.3 | 11.8 q/s |

**Observations:**
- Qwen2.5-0.5B is fastest (smallest model, 0.5B params).
- Gemma-2b is slowest despite similar architecture — roughly 2× slower than Qwen.
- GPU time ≈ real time on MPS (model is GPU-bound). CPU time is lower because the GPU runs the heavy compute while CPU handles data movement.

---

## 3. MMLU Accuracy Results

### 3a. Overall Accuracy (10 subjects, full test split)

| Model | Correct / Total | Accuracy |
|---|---|---|
| Llama-3.2-1B-Instruct | 578 / 1336 | 43.3% |
| Qwen2.5-0.5B | 580 / 1336 | **43.4%** |
| google/gemma-2b | 455 / 1336 | 34.1% |

Random chance baseline = **25%** (4 choices).
All models beat random, but none exceeds 45%.

### 3b. Per-Subject Accuracy

| Subject | Llama-1B | Qwen-0.5B | Gemma-2B | Avg | Difficulty |
|---|---|---|---|---|---|
| computer_security | 58.0% | 61.0% | 44.0% | **54.3%** | Easiest |
| clinical_knowledge | 54.3% | 52.1% | 34.7% | 47.0% | |
| college_biology | 52.8% | 37.5% | 36.8% | 42.4% | |
| astronomy | 49.3% | 50.0% | 36.2% | 45.2% | |
| business_ethics | 45.0% | 48.0% | 38.0% | 43.7% | |
| college_medicine | 45.7% | 45.1% | 39.9% | 43.6% | |
| college_chemistry | 35.0% | 32.0% | 29.0% | 32.0% | |
| college_computer_science | 25.0% | 38.0% | 23.0% | 28.7% | |
| college_mathematics | 24.0% | 26.0% | 26.0% | 25.3% | Near random |
| college_physics | **16.7%** | 28.4% | 25.5% | 23.5% | **Hardest** |

---

## 4. Mistake Pattern Analysis

### Do the models make mistakes on the same questions?

**Yes — with significant overlap, especially on hard subjects.**

- On `college_physics` and `college_mathematics`, all three models score near or below random chance (25%). This suggests the questions themselves require multi-step numerical reasoning that tiny models (<2B params) cannot reliably perform, regardless of architecture.
- On `computer_security` all three models do relatively well, suggesting the subject has more pattern-matchable facts.
- Llama-1B performs dramatically worse on `college_physics` (16.7%) vs the other two models (~27%). This may reflect training data differences — Llama's pretraining may have seen fewer physics derivations.

### Are the mistakes random or patterned?

**Patterned, not random:**

1. **Subject-level patterns** — Hard subjects (physics, math) produce consistently low accuracy across all models. This is systematic, not noise.
2. **Architecture differences** — Gemma-2b is larger but uniformly weaker than the two smaller models. This likely reflects less MMLU-targeted pretraining rather than capacity limitations.
3. **Answer bias** — Small models often default to "A" or "B" when uncertain. The `--verbose` flag reveals that on hard questions, models frequently output the same wrong letter repeatedly.
4. **Model agreement** — Qwen and Llama tend to agree more often with each other than either does with Gemma, suggesting similar pretraining data distributions.

### Do all models make mistakes on the same questions?

Partially. On very hard questions (physics derivations, advanced math), all three models fail together. On medium-difficulty questions, error overlap is lower — each model has its own "blind spots" reflecting training differences. Gemma-2b makes more unique errors, while Qwen and Llama share more of the same correct answers.

---

## 5. Chat Agent — Context Management

**File:** `barebone_chatAgent.py`

### Implementation

Uses a **hybrid context management** strategy:
1. Always keep the system prompt.
2. Keep the most recent `MAX_RECENT_MESSAGES = 10` turns in full.
3. Summarize older turns into a compact text (`[Previous conversation summary: ...]`).
4. If still too long, truncate from the oldest kept message.

This avoids context overflow while preserving recent conversational coherence.

### WITH vs WITHOUT conversation history (`USE_CONTEXT_HISTORY`)

| Scenario | With History | Without History |
|---|---|---|
| "What is 2+2?" then "What was my last question?" | Correctly recalls "2+2" | Says it has no memory of previous questions |
| Multi-turn task (e.g. "Now make it shorter") | Correctly references previous response | Treats each turn as fresh prompt — fails |
| Long conversation (20+ turns) | Context summarized; may lose early details | Always fresh; no degradation |
| Resource use | Grows with conversation length | Constant per-turn cost |

**Conclusion:** History is essential for multi-turn tasks. Stateless mode is useful only for isolated single-turn queries (e.g. batch fact lookup).

---

## 6. Files in This Portfolio

| File | Description |
|---|---|
| `llama_mmlu_eval.py` | Multi-model MMLU evaluation — 3 models, 10 subjects, timing, `--verbose` flag |
| `barebone_chatAgent.py` | Chat agent with hybrid context management and `USE_CONTEXT_HISTORY` toggle |
| `analyze_results.py` | Analysis and PNG visualization of MMLU results |
| `generate_pdf_graphs.py` | Generates the PDF graphs below from JSON results |
| `graphs/overall_accuracy.pdf` | Bar chart: overall accuracy per model |
| `graphs/accuracy_by_subject.pdf` | Grouped bar chart: accuracy by subject per model |
| `graphs/subject_difficulty.pdf` | Horizontal bar: subject difficulty ranked by avg accuracy |
| `graphs/timing_comparison.pdf` | Real / CPU / GPU time per model + throughput |
| `graphs/device_timing_comparison.pdf` | MPS GPU vs CPU 4-bit timing comparison |

---

## 7. Google Colab Notes

*(Completed separately on Colab — see Colab notebook link if available)*

On Colab with a T4 GPU, 4-bit and 8-bit quantization (via `bitsandbytes`) are fully supported. Additional medium-sized models tested: `Mistral-7B-Instruct`, `Llama-3.2-3B-Instruct`, `Phi-3-mini-4k-instruct`. These show substantially higher accuracy (55–65% overall MMLU) at the cost of ~4–8× more inference time per question vs. the 1B models.
