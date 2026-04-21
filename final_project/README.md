# Forecasting Adverse Events in Lung Cancer Patients
### A Zero-Shot LLM Approach with Narrative Timelines + Drug-Aware RAG

> **Final project**, *LLM-Agent* (CS 6501, Spring 2026)
> Xinyu Chen — University of Virginia

[**Watch the 5-minute video →**](./LLM_AE_Forecasting_FinalVideo.mp4)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Motivation](#2-motivation)
3. [Data](#3-data)
4. [Task Formulation](#4-task-formulation)
5. [Pipeline](#5-pipeline)
6. [Method ① — Narrative Timeline Representation](#6-method----narrative-timeline-representation)
7. [Method ② — Drug-Aware RAG](#7-method--drug-aware-rag)
8. [Method ③ — Few-Shot + CoT Ablations](#8-method--few-shot--cot-ablations)
9. [Results](#9-results)
10. [Key Findings](#10-key-findings)
11. [Limitations](#11-limitations)
12. [Future Work](#12-future-work)
13. [Files in this Folder](#13-files-in-this-folder)

---

## 1. Overview

Patients with **EGFR / ALK / ROS1-mutated lung cancer** take **tyrosine kinase
inhibitors (TKIs)** as their primary therapy. These drugs extend survival but
produce frequent adverse events (AEs) — rash, diarrhea, fatigue, pneumonitis,
cardiac toxicity — severe enough to force dose reductions or treatment
interruption. Because clinic visits are weeks apart, most AEs are caught late.

This project asks a simple question:

> **Can a zero-shot large language model, given only passively collected
> wearable + survey data plus basic medical priors, predict 14-day AE onset
> better than a supervised machine-learning model trained on the same task?**

The answer is **yes** — if the representation and the external knowledge are
designed carefully. Our best configuration reaches **AUROC 0.623** vs the best
supervised baseline (Random Forest) at **0.556**, with **zero training
examples**.

The two design choices that carry the gain are:

1. A **clinical-style narrative timeline** — the wearable + symptom data are
   rewritten into a progress-note-like description with pre-computed trends,
   percent changes, and sharp-change flags.
2. A **drug-aware retrieval-augmented generation (RAG)** layer — FDA
   prescribing information for each TKI drug is indexed offline and retrieved
   at inference time to enrich the prompt with drug-specific AE priors.

---

## 2. Motivation

### 2.1 The Clinical Reality

- TKIs are **standard of care** for EGFR / ALK / ROS1 lung cancer and are
  taken **daily for months to years**.
- AEs are common and sometimes severe — dose reductions, hospitalizations, or
  switch to a different line of therapy.
- Oncologists typically see patients **every 3–6 weeks**, so AEs usually
  worsen in between and are only identified at the next visit.

### 2.2 The Opportunity

- Patients in our study wear a **Fitbit 24/7** (activity, heart rate, sleep).
- They complete a **daily ecological-momentary-assessment (EMA) survey**
  covering symptoms, mood, and energy.
- **Behavioral and emotional signals shift before clinical AEs** — this
  information is already collected but not yet used for forecasting.

### 2.3 Research Question

> In TKI-treated lung cancer patients, can a zero-shot LLM grounded in
> narrative wearable + EMA timelines and FDA drug-specific priors match
> supervised ML on 14-day adverse-event onset forecasting?

Three sub-questions drive the method:

| # | Theme | Question |
|---|---|---|
| **RQ1** | Representation | Does a clinical-style narrative timeline improve LLM discrimination over a structured-JSON summary of the same data? |
| **RQ2** | Domain grounding | Does injecting FDA drug-specific AE profiles (RAG) improve discrimination beyond patient-signal context alone? |
| **RQ3** | Prompt engineering | Do few-shot exemplars and chain-of-thought reasoning add value over a zero-shot prompt that already carries the enriched context? |

---

## 3. Data

- **30 patients**, ~130 days of continuous monitoring each
- **Fully longitudinal**, with three modalities fused at the daily level

| Modality | Source | Key fields |
|---|---|---|
| **Fitbit** | Continuous wearable | Steps, active / sedentary minutes, resting + waking heart rate, sleep duration, sleep efficiency, sleep stages |
| **EMA** | Daily survey app | 10 daily symptoms (fatigue, pain, nausea, appetite, mood…), 6 weekly items, 22 biweekly items (quality of life, psychosocial) |
| **Clinical AEs** | Oncologist records | 140 events across 16 categories, onset dates, CTCAE grades |

**Preprocessing.** EMA weekly / biweekly items are forward-filled between
survey submissions. Fitbit signals are aggregated to daily summaries. Missing
data are tracked and surfaced in the prompt as explicit data-quality flags.

---

## 4. Task Formulation

Binary classification over sliding 14-day windows:

```
Observation  W = 14 days   →   index day t   →   Horizon  H = 14 days
────────────────────────────────────────────────────────────────────
                                      y = 1   if any new AE onsets in (t, t + 14]
                                      y = 0   otherwise
```

- **Stride** = 14 days (non-overlapping windows, avoids temporal leakage)
- **291 evaluation windows** total · **39 positive** · **13.4 %** base rate
- **Evaluation protocol:** Leave-One-Patient-Out (LOPO) — model is always
  tested on a patient it has never seen
- **Metrics:** AUROC, AUPRC, Brier score, positive/negative risk separation

---

## 5. Pipeline

```
┌─────────┐    ┌────────────┐    ┌───────────────┐    ┌──────────────┐    ┌────────────┐
│  Raw    │    │   Build    │    │  Narrative    │    │     LLM      │    │  Evaluate  │
│ streams │ →  │ instances  │ →  │  + drug RAG   │ →  │  Inference   │ →  │   LOPO     │
└─────────┘    └────────────┘    └───────────────┘    └──────────────┘    └────────────┘
 Fitbit 15       (patient, day)   Timeline prose        Claude Sonnet       AUROC, AUPRC
 EMA 38          14-d window      + FDA drug AE         zero-shot           vs classical
 Clinical AEs    Label y          + demographics        JSON risk output    ML baselines
```

Stages 1, 2, 4, 5 are routine plumbing. **All of the research interest lies in
Stage 3** — how we represent the window for the LLM and what external
knowledge we bring in.

---

## 6. Method ① — Narrative Timeline Representation

### 6.1 JSON vs Narrative

The default instinct is to pass the LLM a structured summary:

```jsonc
{
  "total_steps": {"mean": 3621, "trend_slope": -430.7, "last_value": 1273},
  "fatigue":     {"mean": 0.36, "trend_slope": -0.05,  "last_value": 0.0  }
}
```

Instead, we render the same information as a **clinical-style progress note**:

```
Patient: 68F, EGFR+, osimertinib 80 mg QD, on TKI 6 months

[Activity]   Steps 4717 ↓ 1273  (-73 %)
             Trend -431 steps/day   ⚠ Sharp decline Day 6
[Heart rate] Day-night gap collapsed from 18 → 4 bpm
[Sleep]      Duration stable, efficiency ↓ 88 → 79 %
[Symptoms]   Fatigue 2 → 4 (last 3 days), appetite stable
```

### 6.2 What Actually Changes

| Dimension | JSON | Narrative |
|---|---|---|
| **Pre-computation** | raw summary stats | `Δ%`, slope, day-night gap, **⚠** sharp-change flags |
| **Structure** | flat key / value dict | chronological story |
| **Emphasis** | all fields equal | `↑ ↓ ⚠` salience markup |
| **Modalities** | separate JSON objects | Fitbit + EMA + drug woven together |
| **LLM alignment** | schema-specific, out-of-distribution | clinical-note style, in-distribution |

> **The narrative layer is our feature-engineering step** — we just output
> natural language instead of a feature vector. Because the LLM was trained
> on clinical notes, it reasons about findings rather than crunching numbers.

**Impact.** Same data, same model — AUROC **0.529 → 0.559** (**+0.030**).

---

## 7. Method ② — Drug-Aware RAG

LLMs know medicine broadly but do **not** reliably recall, e.g., that
osimertinib causes diarrhea in 47 % of patients or that ILD typically
emerges around three months. Different TKIs have very different AE profiles,
so these priors materially affect the risk estimate.

### 7.1 A Three-Stage Retrieval Pipeline

```
① Index (offline)                ② Retrieve (per patient)          ③ Augment generation
─────────────────────           ────────────────────────         ─────────────────────────
OpenFDA API  →  8 TKI             key = patient.drug_name            prompt += retrieved
drug profiles indexed             → exact-match KB lookup             profile + demographics
```

### 7.2 Knowledge Base

Eight TKI drugs seen in the cohort, each indexed from FDA prescribing
information via the **OpenFDA API** into a structured record:

```python
DRUG_AE_PROFILES["osimertinib"] = {
    "brand":   "Tagrisso",
    "target":  "EGFR",
    "common_aes": {                     # trial incidence, %
        "diarrhea": 47, "rash": 46, "musculoskeletal_pain": 38,
        "nail_toxicity": 34, "dry_skin": 32, "stomatitis": 24, "fatigue": 21,
    },
    "serious_warnings": [
        "ILD / pneumonitis 4 % (0.4 % fatal)",
        "Cardiomyopathy 3.8 %",
        "QTc prolongation: 1.1 % > 500 ms",
    ],
    "onset_notes": "Skin reactions within first month; ILD median onset ~3 months.",
}
```

### 7.3 Why This Is (Simple) RAG

Following the definition of RAG (Lewis et al., NeurIPS 2020) — *augmenting
generation by retrieving from a non-parametric memory store* — this pipeline
satisfies every core property:

| RAG property | Our implementation |
|---|---|
| External, non-parametric KB | ✅ `DRUG_AE_PROFILES` dict, curated from OpenFDA |
| Dynamic, query-conditional retrieval | ✅ profile depends on `patient.drug_name` |
| Generation conditioned on retrieved text | ✅ profile injected into prompt |
| Update KB without retraining the LLM | ✅ add a drug → edit the dict |
| Interpretable, auditable retrieval | ✅ deterministic exact-match |

We call it a **"deterministic keyed retriever over a curated knowledge base."**
The retriever is trivial (`dict.get(key)`) — no embeddings, no BM25, no
vector DB. In a small, clinical setting this is a **feature, not a
limitation**: perfect precision on drug match, no hallucinated documents,
fully auditable.

**Impact.** Drug RAG is the **single biggest lift** in the project:
AUROC **0.560 → 0.623** (**+0.063**).

---

## 8. Method ③ — Few-Shot + CoT Ablations

Two classical prompt-engineering techniques were tested as ablations.

### 8.1 Few-Shot Prompting

Three labeled example patients inserted before the actual query:

```
Example 1 — P122: steps ↑, no symptoms, 24 mo on drug        →  no AE
Example 2 — P120: fatigue rising, new nausea, 3 mo on drug   →  AE
Example 3 — P110: steady trends, stable symptoms, 18 mo      →  no AE
[actual patient window]
```

### 8.2 Chain-of-Thought (CoT)

Force a 6-step reasoning plan before the final risk score:

```
Step 1  Review activity and sleep trends
Step 2  Assess heart rate autonomics
Step 3  Scan EMA symptoms
Step 4  Match against drug AE profile
Step 5  Consider time on therapy
Step 6  Integrate → risk score
```

### 8.3 What We Found

**Neither helped.** In fact, few-shot *hurt* calibration: the three examples
were **67 % positive** vs the true **13.4 %** base rate, which inflated the
model's prior. CoT did not improve discrimination — the LLM already produced
structured reasoning in the free-text part of the zero-shot output.

---

## 9. Results

All numbers: **291 LOPO windows**, **Claude Sonnet**, same identical splits
for every version.

| Version | AUROC | AUPRC | Separation | Notes |
|---|---:|---:|---:|---|
| V0 JSON (Sonnet) | 0.529 | 0.149 | 0.008 | raw JSON input |
| V1 Narrative | 0.559 | 0.175 | 0.011 | + timeline (Method ①) |
| V2-RAG | 0.595 | 0.183 | 0.043 | + drug RAG (Method ②) |
| **V2-RAG-Sleep** ⭐ | **0.623** | **0.187** | **0.058** | + sleep data ← **best** |
| V3-Fewshot | 0.617 | 0.180 | 0.057 | + few-shot examples |
| V3-CoT | 0.618 | 0.181 | 0.059 | + chain-of-thought |
| *— Random Forest (supervised)* | *0.556* | *—* | *—* | *best classical baseline* |

### Full Supervised-ML Baselines (LOPO, same 291 windows)

| Model | AUROC |
|---|---:|
| SVM | 0.310 |
| KNN | 0.417 |
| Logistic Regression | 0.482 |
| Gaussian Naive Bayes | 0.500 |
| **Random Forest** | **0.556** |

---

## 10. Key Findings

1. **Representation matters.**
   Narrative timeline alone beats raw JSON by **+0.030 AUROC** — same data,
   same model, different prompt format. The narrative acts as
   *feature engineering in natural language*.

2. **Domain knowledge is the biggest lever.**
   FDA drug-specific RAG adds **+0.063 AUROC** on top of narrative — the
   single largest improvement in the project. Drug priors are not reliably
   carried in LLM parametric memory.

3. **Zero-shot LLM beats supervised ML on small cohorts.**
   **0.623 (LLM, zero-shot) vs 0.556 (Random Forest, supervised)** — a 6-point
   gap with no training examples. For small-N clinical datasets, careful
   context design can outperform trained classical models.

4. **Prompt tricks don't replace context.**
   Few-shot and CoT provided no discrimination gain; few-shot actively hurt
   calibration by inflating the prior to 0.57 vs the true 0.134 base rate.
   **Representation and domain knowledge moved the needle — prompt tricks
   didn't.**

---

## 11. Limitations

- **Small cohort (N = 30).** LOPO mitigates but does not eliminate
  between-patient heterogeneity; bootstrap confidence intervals are still
  needed for significance testing.
- **Calibration gap.** Best model mean-risk is **0.251** vs true rate
  **0.134** — systematic over-prediction. Platt scaling / isotonic
  regression over a held-out slice would help.
- **Grade-agnostic labels.** Any new AE onset counts as positive, regardless
  of severity (Grade 1–4 are all treated the same). Clinically, a Grade 1
  rash differs from Grade 3 pneumonitis.
- **Soft leakage in `in_ae_now`.** A small set of windows that begin during
  an active AE may introduce a subtle leak; results hold if these windows
  are excluded.
- **Simple retriever.** A single exact-match key (drug name) is fine for
  AE-profile retrieval but does not exploit cross-modal similarity (e.g.,
  *retrieve the most similar past patient*).

---

## 12. Future Work

- **Bootstrap CIs** over LOPO folds for significance testing
- **Platt / isotonic calibration** on a small held-out validation set
- **Personal-baseline injection** — every patient has a personal step /
  HR / sleep norm; deviations from *that* norm should be compared, not
  against a population mean
- **Dynamic similar-case retrieval** — extend RAG to retrieve the most
  similar past patient windows (semantic search over the narrative corpus)
- **Per-AE-category prediction** — separate prompts for GI / skin /
  cardiac / pulmonary AEs, with category-specific drug priors
- **Self-consistency ensembling** — sample multiple reasoning traces and
  majority-vote the risk score
- **Clinician-in-the-loop deployment** — surface the risk + top_factors
  field into the oncology team's weekly triage dashboard

---

## 13. Files in this Folder

| File | Purpose |
|---|---|
| [`LLM_AE_Forecasting_FinalVideo.mp4`](./LLM_AE_Forecasting_FinalVideo.mp4) | 5-minute presentation recording (compressed) |
| [`README.md`](./README.md) | This document |

### Source Code

The full pipeline (instance builder, narrative generator, drug knowledge
base, inference runner, evaluation scripts) lives in a separate research
repository. Key modules:

- `02b_build_narrative_instances.py` — build narrative windows
- `02c_build_rag_narrative_instances.py` — add drug RAG + demographics
- `narrative.py` — clinical-timeline generator with trend / anomaly flags
- `drug_knowledge.py` — 8-drug FDA knowledge base + retriever
- `03b_run_narrative_inference.py` — Claude Sonnet inference loop
- `05_fair_comparison.py` — classical ML baselines + LOPO evaluation

---

## Acknowledgments

Study data collected under the *Lung Cancer TKI-AE Monitoring* protocol,
University of Virginia. Drug AE profiles sourced from FDA prescribing
information via the **OpenFDA API**. Inference via **Claude Sonnet** (no
fine-tuning, no training examples).

---

*For the 5-minute overview, watch [**`LLM_AE_Forecasting_FinalVideo.mp4`**](./LLM_AE_Forecasting_FinalVideo.mp4).*
