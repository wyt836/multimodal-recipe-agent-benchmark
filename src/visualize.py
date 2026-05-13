"""Visualise benchmark results.

Reads eval/results/{per_query_results.csv, failure_analysis.csv} and writes
four PNGs to eval/results/figures/ suitable for inclusion in the report:

  1. task_success_heatmap.png — 7 systems × 4 families, success-rate heatmap
  2. cost_vs_quality.png      — token cost vs overall task-success scatter
  3. failure_taxonomy.png     — stacked-bar failure-class breakdown per system
  4. ablation_delta.png       — per-family deltas of V3 ablations vs full V3

Run: `py src/visualize.py`
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from config import FAILURES_CSV, PER_QUERY_CSV, RESULTS_DIR


FIGURES_DIR = RESULTS_DIR / "figures"

SYSTEMS_ORDER = [
    "V0", "V1", "V2", "V3",
    "V3-no-memory", "V3-no-planner", "V3-no-verifier",
]
FAMILIES_ORDER = ["factual", "cross_modal", "multi_hop", "conversational"]
FAMILY_LABEL = {
    "factual":        "Factual",
    "cross_modal":    "Cross-modal",
    "multi_hop":      "Multi-hop",
    "conversational": "Conversational",
}
FAILURE_ORDER = [
    "retrieval", "query_grounding", "grounding", "reasoning", "citation",
]


# ============================================================== Loaders
def _load():
    per_q   = pd.read_csv(PER_QUERY_CSV)
    fails   = pd.read_csv(FAILURES_CSV) if FAILURES_CSV.exists() else pd.DataFrame()
    return per_q, fails


# ============================================================== Plot 1: Heatmap
def plot_heatmap(per_q: pd.DataFrame) -> None:
    """Task-success rate per (system, family). White=0, dark=1."""
    pivot = (per_q.groupby(["system", "family"])["task_success"].mean()
                  .unstack("family")
                  .reindex(index=SYSTEMS_ORDER, columns=FAMILIES_ORDER))

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="Blues", aspect="auto")

    ax.set_xticks(range(len(FAMILIES_ORDER)))
    ax.set_xticklabels([FAMILY_LABEL[f] for f in FAMILIES_ORDER])
    ax.set_yticks(range(len(SYSTEMS_ORDER)))
    ax.set_yticklabels(SYSTEMS_ORDER)

    # numeric annotations
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            color = "white" if v > 0.55 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=10)

    ax.set_title("Task Success Rate by System × Query Family")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Success rate (0–1)")
    plt.tight_layout()
    out = FIGURES_DIR / "task_success_heatmap.png"
    plt.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[viz] wrote {out.name}")


# ============================================================== Plot 2: Cost vs quality
def plot_cost_vs_quality(per_q: pd.DataFrame) -> None:
    """Scatter: total tokens (input+output) vs overall task success."""
    per_q = per_q.copy()
    per_q["total_tokens"] = per_q["tokens_input"].fillna(0) + per_q["tokens_output"].fillna(0)

    agg = (per_q.groupby("system")
                .agg(success=("task_success", "mean"),
                     tokens=("total_tokens", "mean"),
                     latency=("latency_seconds", "mean"))
                .reindex(SYSTEMS_ORDER))

    fig, ax = plt.subplots(figsize=(7, 5))
    palette = {
        "V0": "#888888", "V1": "#1f77b4", "V2": "#2ca02c", "V3": "#d62728",
        "V3-no-memory":   "#ff7f0e",
        "V3-no-planner":  "#9467bd",
        "V3-no-verifier": "#8c564b",
    }
    for name, row in agg.iterrows():
        ax.scatter(row["tokens"], row["success"],
                   s=140, color=palette.get(name, "black"),
                   edgecolor="black", linewidth=0.8, label=name, zorder=3)
        ax.annotate(name, (row["tokens"], row["success"]),
                    xytext=(8, 4), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Mean tokens per query (input + output)")
    ax.set_ylabel("Mean Task Success rate")
    ax.set_title("Quality vs Cost Trade-off")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xscale("log")
    ax.grid(alpha=0.3, zorder=0)
    plt.tight_layout()
    out = FIGURES_DIR / "cost_vs_quality.png"
    plt.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[viz] wrote {out.name}")


# ============================================================== Plot 3: Failure taxonomy
def plot_failure_taxonomy(fails: pd.DataFrame) -> None:
    """Stacked bar: failure count per system, coloured by 5-class taxonomy."""
    if fails.empty:
        print("[viz] no failures to plot")
        return

    counts = (fails.groupby(["system", "failure_type"]).size()
                   .unstack("failure_type", fill_value=0)
                   .reindex(index=SYSTEMS_ORDER, fill_value=0)
                   .reindex(columns=FAILURE_ORDER, fill_value=0))

    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(counts))
    colors = ["#d62728", "#ff7f0e", "#1f77b4", "#9467bd", "#2ca02c"]
    for ftype, color in zip(FAILURE_ORDER, colors):
        vals = counts[ftype].values
        ax.bar(counts.index, vals, bottom=bottom, color=color, label=ftype,
               edgecolor="white", linewidth=0.5)
        bottom = bottom + vals

    ax.set_ylabel("Number of failed queries (out of 20)")
    ax.set_title("Failure Mode Distribution (5-class Taxonomy)")
    ax.set_ylim(0, 20)
    ax.legend(title="Failure type", loc="upper right", fontsize=9, framealpha=0.9)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / "failure_taxonomy.png"
    plt.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[viz] wrote {out.name}")


# ============================================================== Plot 4: Ablation delta
def plot_ablation_delta(per_q: pd.DataFrame) -> None:
    """For each ablation, show task-success delta vs full V3, per family."""
    pivot = (per_q.groupby(["system", "family"])["task_success"].mean()
                  .unstack("family")
                  .reindex(columns=FAMILIES_ORDER))

    if "V3" not in pivot.index:
        print("[viz] V3 baseline missing, skipping ablation_delta")
        return

    baseline = pivot.loc["V3"]
    ablations = ["V3-no-memory", "V3-no-planner", "V3-no-verifier"]
    delta = pivot.loc[ablations] - baseline

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(FAMILIES_ORDER))
    width = 0.25
    colors = {"V3-no-memory": "#ff7f0e", "V3-no-planner": "#9467bd", "V3-no-verifier": "#8c564b"}
    for i, abl in enumerate(ablations):
        ax.bar(x + (i - 1) * width, delta.loc[abl].values, width,
               label=abl, color=colors[abl], edgecolor="black", linewidth=0.5)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABEL[f] for f in FAMILIES_ORDER])
    ax.set_ylabel("Task Success Δ vs full V3")
    ax.set_title("Ablation Impact — Negative Means Removing the Component Hurt V3")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = FIGURES_DIR / "ablation_delta.png"
    plt.savefig(out, dpi=180)
    plt.close(fig)
    print(f"[viz] wrote {out.name}")


# ============================================================== Main
def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    per_q, fails = _load()
    if per_q.empty:
        print("[viz] per_query_results.csv is empty; run the benchmark first.")
        return

    plot_heatmap(per_q)
    plot_cost_vs_quality(per_q)
    plot_failure_taxonomy(fails)
    plot_ablation_delta(per_q)

    print(f"[done] figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
