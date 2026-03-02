"""
Generate PDF graphs for the Running an LLM portfolio.
Uses the most recent multi-model MMLU results.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Load data ──────────────────────────────────────────────────────────────────
RESULTS_FILE = Path(__file__).parent.parent / "multi_model_mmlu_results_20260114_154605.json"
OUTPUT_DIR   = Path(__file__).parent / "graphs"
OUTPUT_DIR.mkdir(exist_ok=True)

with open(RESULTS_FILE) as f:
    data = json.load(f)

model_results = data["model_results"]

# Short names for display
def short_name(full_name):
    return full_name.split("/")[-1]

models   = [short_name(r["model_name"]) for r in model_results]
subjects = [sr["subject"] for sr in model_results[0]["results"]]

# ── 1. Overall Accuracy ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
accuracies = [r["overall_accuracy"] for r in model_results]
colors = ["#4C72B0", "#DD8452", "#55A868"]
bars = ax.bar(models, accuracies, color=colors, width=0.5, edgecolor="white", linewidth=1.2)

for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{acc:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylabel("Overall Accuracy (%)", fontsize=12)
ax.set_title("Overall MMLU Accuracy — 3 Models × 10 Subjects\n(Apple MPS, FP16, 1 336 questions each)",
             fontsize=12, fontweight="bold")
ax.set_ylim(0, 60)
ax.axhline(25, color="gray", linestyle="--", linewidth=0.8, label="Random chance (25%)")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "overall_accuracy.pdf", dpi=150)
plt.close(fig)
print("Saved: overall_accuracy.pdf")

# ── 2. Accuracy by Subject ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(subjects))
width = 0.25

for i, (mr, color) in enumerate(zip(model_results, colors)):
    accs = [sr["accuracy"] for sr in mr["results"]]
    ax.bar(x + i * width, accs, width, label=short_name(mr["model_name"]),
           color=color, edgecolor="white", linewidth=0.8)

ax.set_xlabel("Subject", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("MMLU Accuracy by Subject — 3 Models (Apple MPS, FP16)",
             fontsize=13, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels([s.replace("_", "\n") for s in subjects], fontsize=9)
ax.legend(fontsize=10)
ax.axhline(25, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Random chance")
ax.set_ylim(0, 80)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "accuracy_by_subject.pdf", dpi=150)
plt.close(fig)
print("Saved: accuracy_by_subject.pdf")

# ── 3. Subject Difficulty (average accuracy across models) ─────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
avg_accs = []
std_accs = []
for j, subject in enumerate(subjects):
    vals = [mr["results"][j]["accuracy"] for mr in model_results]
    avg_accs.append(np.mean(vals))
    std_accs.append(np.std(vals))

sorted_idx = np.argsort(avg_accs)
sorted_subjects = [subjects[i].replace("_", "\n") for i in sorted_idx]
sorted_avgs     = [avg_accs[i] for i in sorted_idx]
sorted_stds     = [std_accs[i] for i in sorted_idx]

bar_colors = ["#d62728" if a < 30 else "#2ca02c" if a > 50 else "#4C72B0"
              for a in sorted_avgs]
bars = ax.barh(sorted_subjects, sorted_avgs, xerr=sorted_stds,
               color=bar_colors, edgecolor="white", linewidth=0.8,
               error_kw={"capsize": 4, "elinewidth": 1.2})

ax.axvline(25, color="gray", linestyle="--", linewidth=0.9, label="Random chance (25%)")
ax.set_xlabel("Average Accuracy (%) ± std across models", fontsize=11)
ax.set_title("Subject Difficulty — Average Accuracy Across All 3 Models",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "subject_difficulty.pdf", dpi=150)
plt.close(fig)
print("Saved: subject_difficulty.pdf")

# ── 4. Timing Comparison ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Bar chart: real vs cpu vs gpu time per model
timing_keys = ["real_time", "cpu_time", "gpu_time"]
timing_labels = ["Real Time", "CPU Time", "GPU Time (MPS)"]
x = np.arange(len(models))
w = 0.22

for k, (key, label, color) in enumerate(zip(timing_keys, timing_labels, ["#4C72B0", "#DD8452", "#55A868"])):
    vals = [r["timings"][key] for r in model_results]
    axes[0].bar(x + k * w, vals, w, label=label, color=color, edgecolor="white")

axes[0].set_xticks(x + w)
axes[0].set_xticklabels(models, fontsize=10)
axes[0].set_ylabel("Time (seconds)", fontsize=11)
axes[0].set_title("Inference Time per Model\n(MPS GPU, FP16, 1 336 questions)", fontsize=11, fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].grid(axis="y", alpha=0.3)

# Bar chart: questions per second throughput
for k, (mr, color) in enumerate(zip(model_results, colors)):
    rt = mr["timings"]["real_time"]
    qps = mr["total_questions"] / rt
    axes[1].bar(k, qps, color=color, edgecolor="white", label=short_name(mr["model_name"]))
    axes[1].text(k, qps + 0.2, f"{qps:.1f} q/s", ha="center", va="bottom", fontsize=10, fontweight="bold")

axes[1].set_xticks(range(len(models)))
axes[1].set_xticklabels(models, fontsize=10)
axes[1].set_ylabel("Throughput (questions / second)", fontsize=11)
axes[1].set_title("Model Throughput\n(MPS GPU, FP16)", fontsize=11, fontweight="bold")
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "timing_comparison.pdf", dpi=150)
plt.close(fig)
print("Saved: timing_comparison.pdf")

# ── 5. CPU vs GPU timing (using early single-model results) ────────────────────
# From JSON files: MPS 23.5s vs CPU 4-bit 85.2s for 252 questions
fig, ax = plt.subplots(figsize=(9, 5))

setups = [
    "MPS GPU\nNo Quant\n(FP16)",
    "CPU\n4-bit Quant\n(bitsandbytes)",
]
times_252 = [23.54, 85.15]   # seconds for 252 questions (2 subjects)
# Normalize to per-question time
per_q = [t / 252 for t in times_252]

bar_colors2 = ["#55A868", "#DD8452"]
bars = ax.bar(setups, per_q, color=bar_colors2, width=0.4, edgecolor="white", linewidth=1.2)
for bar, t, total in zip(bars, per_q, times_252):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{t:.3f}s/q\n({total:.1f}s total)", ha="center", va="bottom", fontsize=10)

ax.set_ylabel("Time per Question (seconds)", fontsize=12)
ax.set_title("Device Timing Comparison — Llama-3.2-1B on 252 Questions\n"
             "(MacBook: 4-bit GPU quant not available; CPU 4-bit via bitsandbytes)",
             fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "device_timing_comparison.pdf", dpi=150)
plt.close(fig)
print("Saved: device_timing_comparison.pdf")

print(f"\nAll PDF graphs saved to: {OUTPUT_DIR}")
