"""Generate figures for the inference-time strategies report.

Usage:
    python report/generate_strategy_figures.py
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

FIG_DIR = Path("report_strategies/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_DIR = Path("results/benchmark")

STRATEGY_NAMES = ["baseline", "verify", "crop_zoom", "decompose", "best_of_n", "adaptive"]
STRATEGY_LABELS = {
    "baseline": "Baseline",
    "verify": "Verify",
    "crop_zoom": "Crop-Zoom",
    "decompose": "Decompose",
    "best_of_n": "Best-of-5",
    "adaptive": "Adaptive",
}
MODEL_NAMES = ["haiku45", "sonnet46", "baseline", "adaptive"]
MODEL_LABELS = {
    "haiku45": "Haiku 4.5",
    "sonnet46": "Sonnet 4.6",
    "baseline": "Qwen3-VL-8B",
    "adaptive": "Qwen + Adaptive",
}

TASK_ORDER = [
    "counting_grid", "pie_chart", "progress_bar", "colored_paths",
    "nested_squares", "hierarchy_depth", "scatter_plot",
    "realistic_table", "text_degradation",
]

TASK_LABELS = {
    "counting_grid": "Counting Grid",
    "pie_chart": "Pie Chart",
    "progress_bar": "Progress Bar",
    "colored_paths": "Colored Paths",
    "nested_squares": "Nested Squares",
    "hierarchy_depth": "Hierarchy Depth",
    "scatter_plot": "Scatter Plot",
    "realistic_table": "Realistic Table",
    "text_degradation": "Text Degradation",
}

COLORS = {
    "baseline": "#6c757d",
    "verify": "#28a745",
    "crop_zoom": "#fd7e14",
    "decompose": "#6f42c1",
    "best_of_n": "#17a2b8",
    "adaptive": "#dc3545",
}

MODEL_COLORS = {
    "haiku45": "#1976d2",
    "sonnet46": "#d32f2f",
    "baseline": "#6c757d",
    "adaptive": "#ff9800",
}


def load_results(subdir):
    path = BENCHMARK_DIR / subdir / "results.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path)]


def task_accuracy(results, task):
    recs = [r for r in results if r["task_name"] == task]
    if not recs:
        return None
    return sum(1 for r in recs if r.get("correct")) / len(recs) * 100


# ── Figure 1: Strategy comparison bar chart ──────────────────────────────

def fig_strategy_comparison():
    """Grouped bar chart comparing all strategies across tasks."""
    fig, ax = plt.subplots(figsize=(14, 6))

    all_results = {s: load_results(s) for s in STRATEGY_NAMES}

    x = np.arange(len(TASK_ORDER))
    n_strategies = len(STRATEGY_NAMES)
    width = 0.13
    offsets = np.arange(n_strategies) - (n_strategies - 1) / 2

    for i, s in enumerate(STRATEGY_NAMES):
        accs = [task_accuracy(all_results[s], t) or 0 for t in TASK_ORDER]
        bars = ax.bar(x + offsets[i] * width, accs, width,
                      label=STRATEGY_LABELS[s], color=COLORS[s], edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Inference-Time Strategy Comparison on Qwen3-VL-8B Blind Spots", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in TASK_ORDER], rotation=35, ha="right", fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.axhline(y=50, color="#ccc", linestyle="--", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "strategy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved strategy_comparison.png")


# ── Figure 2: Strategy delta chart ───────────────────────────────────────

def fig_strategy_deltas():
    """Horizontal bar chart showing accuracy delta vs baseline for each strategy."""
    fig, ax = plt.subplots(figsize=(10, 5))

    baseline_results = load_results("baseline")
    strategies = ["verify", "best_of_n", "crop_zoom", "decompose", "adaptive"]

    means = {}
    baseline_mean = np.mean([task_accuracy(baseline_results, t) or 0 for t in TASK_ORDER])
    for s in strategies:
        results = load_results(s)
        mean = np.mean([task_accuracy(results, t) or 0 for t in TASK_ORDER])
        means[s] = mean - baseline_mean

    # Sort by delta
    sorted_strategies = sorted(strategies, key=lambda s: means[s])

    y = np.arange(len(sorted_strategies))
    deltas = [means[s] for s in sorted_strategies]
    colors = ["#28a745" if d > 0 else "#dc3545" for d in deltas]

    bars = ax.barh(y, deltas, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([STRATEGY_LABELS[s] for s in sorted_strategies], fontsize=11)
    ax.set_xlabel("Accuracy Change vs Baseline (percentage points)", fontsize=11)
    ax.set_title("Strategy Impact on Mean Accuracy (Qwen3-VL-8B)", fontsize=13, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, delta in zip(bars, deltas):
        ax.text(bar.get_width() + (0.3 if delta >= 0 else -0.3), bar.get_y() + bar.get_height() / 2,
                f"{delta:+.1f}p", va="center", ha="left" if delta >= 0 else "right",
                fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "strategy_deltas.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved strategy_deltas.png")


# ── Figure 3: Cross-model comparison ─────────────────────────────────────

def fig_cross_model():
    """Grouped bar chart comparing Haiku, Sonnet, Qwen, and Qwen+adaptive."""
    fig, ax = plt.subplots(figsize=(14, 6))

    all_results = {m: load_results(m) for m in MODEL_NAMES}

    x = np.arange(len(TASK_ORDER))
    n_models = len(MODEL_NAMES)
    width = 0.18
    offsets = np.arange(n_models) - (n_models - 1) / 2

    for i, m in enumerate(MODEL_NAMES):
        accs = [task_accuracy(all_results[m], t) or 0 for t in TASK_ORDER]
        ax.bar(x + offsets[i] * width, accs, width,
               label=MODEL_LABELS[m], color=MODEL_COLORS[m], edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Cross-Model Comparison on Blind-Spot Tasks (Same 176 Instances)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in TASK_ORDER], rotation=35, ha="right", fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left", fontsize=10)
    ax.axhline(y=50, color="#ccc", linestyle="--", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "cross_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved cross_model_comparison.png")


# ── Figure 4: Heatmap ────────────────────────────────────────────────────

def fig_heatmap():
    """Heatmap of accuracy across all strategies and tasks."""
    fig, ax = plt.subplots(figsize=(10, 6))

    all_results = {s: load_results(s) for s in STRATEGY_NAMES}

    data = []
    for s in STRATEGY_NAMES:
        row = [task_accuracy(all_results[s], t) or 0 for t in TASK_ORDER]
        data.append(row)
    data = np.array(data)

    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(TASK_ORDER)))
    ax.set_xticklabels([TASK_LABELS[t] for t in TASK_ORDER], rotation=40, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(STRATEGY_NAMES)))
    ax.set_yticklabels([STRATEGY_LABELS[s] for s in STRATEGY_NAMES], fontsize=11)

    # Annotate cells
    for i in range(len(STRATEGY_NAMES)):
        for j in range(len(TASK_ORDER)):
            val = data[i, j]
            color = "white" if val < 30 or val > 80 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=9, color=color, fontweight="bold")

    ax.set_title("Strategy × Task Accuracy Heatmap (Qwen3-VL-8B)", fontsize=13, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy (%)", fontsize=11)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "strategy_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved strategy_heatmap.png")


# ── Figure 5: Adaptive routing diagram ───────────────────────────────────

def fig_adaptive_routing():
    """Show per-task best strategy and improvement from adaptive routing."""
    fig, ax = plt.subplots(figsize=(12, 5))

    baseline_results = load_results("baseline")
    adaptive_results = load_results("adaptive")

    tasks = TASK_ORDER
    baseline_accs = [task_accuracy(baseline_results, t) or 0 for t in tasks]
    adaptive_accs = [task_accuracy(adaptive_results, t) or 0 for t in tasks]
    deltas = [a - b for a, b in zip(adaptive_accs, baseline_accs)]

    # Routing table
    routing = {
        "counting_grid": "baseline", "pie_chart": "decompose", "progress_bar": "decompose",
        "colored_paths": "baseline", "nested_squares": "best_of_n",
        "hierarchy_depth": "verify", "scatter_plot": "baseline",
        "realistic_table": "verify", "text_degradation": "baseline",
    }

    x = np.arange(len(tasks))
    width = 0.35

    ax.bar(x - width / 2, baseline_accs, width, label="Baseline", color="#6c757d", edgecolor="white")
    bar_colors = [COLORS.get(routing[t], "#dc3545") for t in tasks]
    ax.bar(x + width / 2, adaptive_accs, width, label="Adaptive", color=bar_colors, edgecolor="white")

    # Annotate deltas
    for i, (ba, aa, d) in enumerate(zip(baseline_accs, adaptive_accs, deltas)):
        if abs(d) >= 3:
            color = "#28a745" if d > 0 else "#dc3545"
            ax.annotate(f"{d:+.0f}p", xy=(i + width / 2, aa + 1), ha="center", fontsize=9,
                        color=color, fontweight="bold")

    # Annotate routing
    for i, t in enumerate(tasks):
        ax.text(i + width / 2, -8, STRATEGY_LABELS[routing[t]], ha="center", fontsize=7,
                color=COLORS.get(routing[t], "black"), fontstyle="italic", rotation=30)

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Adaptive Routing: Per-Task Strategy Selection", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in tasks], rotation=35, ha="right", fontsize=10)
    ax.set_ylim(-12, 105)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "adaptive_routing.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved adaptive_routing.png")


# ── Figure 6: Per-sample Venn-style agreement ─────────────────────────────

def fig_model_agreement():
    """Stacked bar showing per-task agreement between models."""
    fig, ax = plt.subplots(figsize=(12, 5))

    models = {"Haiku 4.5": "haiku45", "Sonnet 4.6": "sonnet46", "Qwen 8B": "baseline"}
    all_results = {}
    for label, subdir in models.items():
        by_id = {}
        for r in load_results(subdir):
            by_id[r["sample_id"]] = r.get("correct", False)
        all_results[label] = by_id

    task_by_id = {}
    for r in load_results("baseline"):
        task_by_id[r["sample_id"]] = r["task_name"]

    categories = ["All correct", "Claude only", "Qwen only", "All wrong"]
    cat_colors = ["#28a745", "#1976d2", "#ff9800", "#dc3545"]

    data = {cat: [] for cat in categories}
    for task in TASK_ORDER:
        ids = [sid for sid, t in task_by_id.items() if t == task]
        n = len(ids)
        all_right = sum(1 for sid in ids if all(all_results[m].get(sid) for m in all_results)) / n * 100
        claude_only = sum(1 for sid in ids if (all_results["Haiku 4.5"].get(sid) or all_results["Sonnet 4.6"].get(sid)) and not all_results["Qwen 8B"].get(sid)) / n * 100
        qwen_only = sum(1 for sid in ids if all_results["Qwen 8B"].get(sid) and not all_results["Haiku 4.5"].get(sid) and not all_results["Sonnet 4.6"].get(sid)) / n * 100
        all_wrong_pct = sum(1 for sid in ids if not any(all_results[m].get(sid) for m in all_results)) / n * 100
        data["All correct"].append(all_right)
        data["Claude only"].append(claude_only)
        data["Qwen only"].append(qwen_only)
        data["All wrong"].append(all_wrong_pct)

    x = np.arange(len(TASK_ORDER))
    bottom = np.zeros(len(TASK_ORDER))
    for cat, color in zip(categories, cat_colors):
        ax.bar(x, data[cat], bottom=bottom, label=cat, color=color, edgecolor="white", width=0.7)
        bottom += np.array(data[cat])

    ax.set_ylabel("Percentage of Samples", fontsize=12)
    ax.set_title("Per-Sample Agreement: Claude vs Qwen (Same Instances)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in TASK_ORDER], rotation=35, ha="right", fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_agreement.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved model_agreement.png")


if __name__ == "__main__":
    print("Generating strategy report figures...")
    fig_strategy_comparison()
    fig_strategy_deltas()
    fig_cross_model()
    fig_heatmap()
    fig_adaptive_routing()
    fig_model_agreement()
    print("Done.")
