"""Generate all figures for the VLM Blind Spots report."""

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
from collections import defaultdict

from analysis import TASK_CLASSIFICATION

RESULTS_FILE = sys.argv[1] if len(sys.argv) > 1 else "results/final_all.jsonl"
FIG_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("report/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Load results ──────────────────────────────────────────────────────────

def load_results():
    tasks = {}
    with open(RESULTS_FILE) as f:
        for line in f:
            r = json.loads(line)
            t = r["task_name"]
            tasks.setdefault(t, {"total": 0, "correct": 0, "records": []})
            tasks[t]["total"] += 1
            tasks[t]["correct"] += 1 if r.get("correct") else 0
            tasks[t]["records"].append(r)
    return tasks

def classify_tasks(tasks):
    """Separate image tasks from text controls using registry."""
    from tasks import TASK_REGISTRY
    text_controls = set()
    for name in TASK_REGISTRY:
        base = name.rsplit("_text", 1)[0] if name.endswith("_text") else None
        if base and base in TASK_REGISTRY:
            text_controls.add(name)

    img = {t: v for t, v in tasks.items() if t not in text_controls}
    txt = {t: v for t, v in tasks.items() if t in text_controls}
    return img, txt

def get_pairs(img_tasks, txt_tasks):
    """Get matched pairs of (image_task, text_control)."""
    pairs = []
    for t in sorted(img_tasks):
        tt = t + "_text"
        if tt in txt_tasks:
            img_acc = img_tasks[t]["correct"] / img_tasks[t]["total"] * 100
            txt_acc = txt_tasks[tt]["correct"] / txt_tasks[tt]["total"] * 100
            gap = txt_acc - img_acc
            cls = TASK_CLASSIFICATION.get(t, {})
            pairs.append({
                "task": t,
                "img_acc": img_acc,
                "txt_acc": txt_acc,
                "gap": gap,
                "perception": cls.get("perception", "?"),
                "reasoning": cls.get("reasoning", "?"),
                "category": cls.get("category", "?"),
                "img_n": img_tasks[t]["total"],
                "txt_n": txt_tasks[tt]["total"],
            })
    return pairs


# ── Figure 1: Main gap chart (sorted by gap) ─────────────────────────────

def fig1_gap_chart(pairs):
    """Horizontal bar chart showing text-image accuracy gap for all tasks."""
    pairs_sorted = sorted(pairs, key=lambda p: p["gap"], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 12))

    tasks = [p["task"].replace("_", " ") for p in pairs_sorted]
    gaps = [p["gap"] for p in pairs_sorted]
    colors = []
    for g in gaps:
        if g > 15:
            colors.append("#e74c3c")  # red = perceptual bottleneck
        elif g < -15:
            colors.append("#3498db")  # blue = reasoning bottleneck
        else:
            colors.append("#95a5a6")  # gray = mixed/minimal

    y_pos = np.arange(len(tasks))
    ax.barh(y_pos, gaps, color=colors, height=0.7, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tasks, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Text − Image Accuracy Gap (percentage points)", fontsize=11)
    ax.set_title("Perception vs Reasoning Diagnostic:\nText-Only Control Accuracy Gap", fontsize=13, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.axvline(x=15, color="#e74c3c", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(x=-15, color="#3498db", linewidth=0.5, linestyle="--", alpha=0.5)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#e74c3c", label="Perceptual bottleneck (gap > +15%)"),
        mpatches.Patch(facecolor="#3498db", label="Reasoning bottleneck (gap < −15%)"),
        mpatches.Patch(facecolor="#95a5a6", label="Mixed / minimal gap"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    # Add gap values as text
    for i, (g, p) in enumerate(zip(gaps, pairs_sorted)):
        offset = 1 if g >= 0 else -1
        ha = "left" if g >= 0 else "right"
        ax.text(g + offset, i, f"{g:+.0f}%", va="center", ha=ha, fontsize=7.5, color="black")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig1_gap_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig1_gap_chart.png")


# ── Figure 2: Image vs Text scatter ──────────────────────────────────────

def fig2_scatter(pairs):
    """Scatter plot: image accuracy vs text accuracy for each task."""
    fig, ax = plt.subplots(figsize=(8, 8))

    for p in pairs:
        color = "#e74c3c" if p["gap"] > 15 else "#3498db" if p["gap"] < -15 else "#95a5a6"
        ax.scatter(p["txt_acc"], p["img_acc"], c=color, s=60, edgecolors="white", linewidth=0.5, zorder=3)
        # Label worst performers
        if p["gap"] > 40 or p["gap"] < -15 or p["img_acc"] < 30:
            ax.annotate(p["task"].replace("_", "\n"), (p["txt_acc"], p["img_acc"]),
                       fontsize=6.5, ha="center", va="bottom",
                       xytext=(0, 6), textcoords="offset points")

    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, linewidth=1, label="y = x (no gap)")
    ax.set_xlabel("Text-Only Control Accuracy (%)", fontsize=11)
    ax.set_ylabel("Image Task Accuracy (%)", fontsize=11)
    ax.set_title("Image vs Text Accuracy per Task", fontsize=13, fontweight="bold")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_aspect("equal")

    # Quadrant labels
    ax.text(95, 5, "Reasoning\nbottleneck", ha="right", va="bottom", fontsize=9, color="#3498db", alpha=0.7)
    ax.text(5, 95, "Text harder\nthan image", ha="left", va="top", fontsize=9, color="#27ae60", alpha=0.7)
    ax.text(95, 95, "Both easy", ha="right", va="top", fontsize=9, color="#95a5a6", alpha=0.7)
    ax.text(5, 5, "Both hard", ha="left", va="bottom", fontsize=9, color="#8e44ad", alpha=0.7)

    # Highlight perceptual zone
    ax.fill_between([50, 105], [0, 0], [50, 50], alpha=0.05, color="#e74c3c")
    ax.text(75, 25, "Perceptual\nbottleneck", ha="center", va="center", fontsize=9, color="#e74c3c", alpha=0.7)

    legend_elements = [
        mpatches.Patch(facecolor="#e74c3c", label="Perceptual bottleneck"),
        mpatches.Patch(facecolor="#3498db", label="Reasoning bottleneck"),
        mpatches.Patch(facecolor="#95a5a6", label="Mixed / minimal"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig2_scatter.png")


# ── Figure 3: Category bar chart ─────────────────────────────────────────

def fig3_category_bars(pairs):
    """Grouped bar chart: image vs text accuracy by task category."""
    cat_data = defaultdict(lambda: {"img": [], "txt": []})
    for p in pairs:
        cat = p["category"]
        cat_data[cat]["img"].append(p["img_acc"])
        cat_data[cat]["txt"].append(p["txt_acc"])

    cats = sorted(cat_data.keys(), key=lambda c: np.mean(cat_data[c]["img"]))
    img_means = [np.mean(cat_data[c]["img"]) for c in cats]
    txt_means = [np.mean(cat_data[c]["txt"]) for c in cats]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(cats))
    w = 0.35

    bars1 = ax.bar(x - w/2, img_means, w, label="Image", color="#e74c3c", alpha=0.8)
    bars2 = ax.bar(x + w/2, txt_means, w, label="Text Control", color="#3498db", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in cats], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy by Task Category: Image vs Text Control", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.2)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}%", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}%", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_category_bars.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig3_category_bars.png")


# ── Figure 4: Perception demand vs actual gap ────────────────────────────

def fig4_perception_demand(pairs):
    """Box plot: actual gap grouped by theoretical perception demand."""
    demand_levels = ["none", "low", "medium", "high"]
    groups = {d: [] for d in demand_levels}
    for p in pairs:
        d = p["perception"]
        if d in groups:
            groups[d].append(p["gap"])

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [groups[d] for d in demand_levels if groups[d]]
    labels = [d for d in demand_levels if groups[d]]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
    colors_box = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    for patch, color in zip(bp["boxes"], colors_box[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay individual points
    for i, (d, vals) in enumerate(zip(labels, data)):
        jitter = np.random.normal(0, 0.05, len(vals))
        ax.scatter([i + 1 + j for j in jitter], vals, color="black", s=20, alpha=0.5, zorder=3)

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.3)
    ax.set_xlabel("Theoretical Perception Demand", fontsize=11)
    ax.set_ylabel("Text − Image Accuracy Gap (%)", fontsize=11)
    ax.set_title("Does Perception Demand Predict the Gap?", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig4_perception_demand.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig4_perception_demand.png")


# ── Figure 5: Top 10 worst blind spots ───────────────────────────────────

def fig5_worst_blindspots(pairs):
    """Paired bar chart showing the 10 worst perceptual blind spots."""
    # Worst = largest positive gap (text >> image)
    worst = sorted(pairs, key=lambda p: p["gap"], reverse=True)[:10]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(worst))
    w = 0.35

    img_vals = [p["img_acc"] for p in worst]
    txt_vals = [p["txt_acc"] for p in worst]

    bars1 = ax.bar(x - w/2, img_vals, w, label="Image", color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + w/2, txt_vals, w, label="Text Control", color="#2ecc71", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([p["task"].replace("_", "\n") for p in worst], rotation=0, ha="center", fontsize=8)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Top 10 Perceptual Blind Spots\n(Largest text > image gap)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.2)

    # Gap annotations
    for i, p in enumerate(worst):
        mid = (p["img_acc"] + p["txt_acc"]) / 2
        ax.annotate(f"Δ{p['gap']:+.0f}%", (i, max(p["img_acc"], p["txt_acc"]) + 3),
                   ha="center", fontsize=8, fontweight="bold", color="#c0392b")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_worst_blindspots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig5_worst_blindspots.png")


# ── Figure 6: Reasoning failures ─────────────────────────────────────────

def fig6_reasoning_failures(pairs):
    """Bar chart for tasks where both image and text fail (reasoning bottleneck)."""
    reasoning = [p for p in pairs if p["txt_acc"] < 50]
    reasoning = sorted(reasoning, key=lambda p: p["img_acc"])

    if not reasoning:
        print("  ⚠ No reasoning failures found, skipping fig6")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(reasoning))
    w = 0.35

    img_vals = [p["img_acc"] for p in reasoning]
    txt_vals = [p["txt_acc"] for p in reasoning]

    ax.bar(x - w/2, img_vals, w, label="Image", color="#e74c3c", alpha=0.85)
    ax.bar(x + w/2, txt_vals, w, label="Text Control", color="#3498db", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([p["task"].replace("_", "\n") for p in reasoning], fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Reasoning Bottlenecks\n(Both image and text accuracy < 50%)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 55)
    ax.grid(axis="y", alpha=0.2)
    ax.axhline(y=25, color="gray", linewidth=0.5, linestyle=":", alpha=0.5, label="Random baseline (MC4)")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig6_reasoning_failures.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig6_reasoning_failures.png")


# ── Figure 7: Heatmap of task classification ──────────────────────────────

def fig7_classification_heatmap(pairs):
    """Heatmap: perception demand × reasoning demand, colored by image accuracy."""
    demands_p = ["none", "low", "medium", "high"]
    demands_r = ["none", "low", "medium", "high"]

    grid = np.full((len(demands_r), len(demands_p)), np.nan)
    grid_labels = [[[] for _ in demands_p] for _ in demands_r]

    for p in pairs:
        pi = demands_p.index(p["perception"]) if p["perception"] in demands_p else -1
        ri = demands_r.index(p["reasoning"]) if p["reasoning"] in demands_r else -1
        if pi >= 0 and ri >= 0:
            if np.isnan(grid[ri][pi]):
                grid[ri][pi] = p["img_acc"]
            else:
                grid[ri][pi] = (grid[ri][pi] + p["img_acc"]) / 2
            grid_labels[ri][pi].append(p["task"])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(demands_p)))
    ax.set_xticklabels(demands_p)
    ax.set_yticks(range(len(demands_r)))
    ax.set_yticklabels(demands_r)
    ax.set_xlabel("Perception Demand", fontsize=11)
    ax.set_ylabel("Reasoning Demand", fontsize=11)
    ax.set_title("Image Accuracy by Demand Classification", fontsize=13, fontweight="bold")

    # Add text annotations
    for ri in range(len(demands_r)):
        for pi in range(len(demands_p)):
            val = grid[ri][pi]
            if not np.isnan(val):
                names = grid_labels[ri][pi]
                names_str = "\n".join(n.replace("_", " ")[:15] for n in names[:3])
                if len(names) > 3:
                    names_str += f"\n+{len(names)-3} more"
                text_color = "white" if val < 50 else "black"
                ax.text(pi, ri, f"{val:.0f}%\n{names_str}", ha="center", va="center",
                       fontsize=6.5, color=text_color)

    plt.colorbar(im, ax=ax, label="Mean Image Accuracy (%)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig7_classification_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig7_classification_heatmap.png")


# ── Figure 8: Example failure images ─────────────────────────────────────

def fig8_failure_examples(tasks_data):
    """Show sample images from worst-performing tasks."""
    from PIL import Image

    worst_tasks = ["colored_paths", "counting_grid", "dense_text", "nested_squares",
                   "color_coded_cells", "edge_crossing", "text_degradation"]

    available = []
    for task_name in worst_tasks:
        if task_name not in tasks_data:
            continue
        # Find a wrong sample
        wrong = [r for r in tasks_data[task_name]["records"] if not r.get("correct")]
        if wrong:
            r = wrong[0]
            img_path = r.get("image_path", "")
            if img_path and os.path.exists(img_path):
                available.append((task_name, r, img_path))

    if not available:
        print("  ⚠ No failure images found, skipping fig8")
        return

    n = min(len(available), 6)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    if rows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__len__') else [row])]

    for i, (task_name, record, img_path) in enumerate(available[:n]):
        ax = axes_flat[i]
        img = Image.open(img_path)
        ax.imshow(img)
        gt = record.get("ground_truth", "?")
        ans = record.get("parsed_answer", "?")
        # Truncate long strings
        if isinstance(gt, str) and len(gt) > 40:
            gt = gt[:37] + "..."
        if isinstance(ans, str) and len(ans) > 40:
            ans = ans[:37] + "..."
        ax.set_title(f"{task_name.replace('_', ' ')}\nGT: {gt} | Model: {ans}", fontsize=9)
        ax.axis("off")

    # Hide unused axes
    for i in range(n, len(axes_flat)):
        axes_flat[i].axis("off")

    fig.suptitle("Example Failures from Worst-Performing Tasks", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig8_failure_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig8_failure_examples.png")


# ── Figure 9: Summary statistics ─────────────────────────────────────────

def fig9_summary_pie(pairs):
    """Pie chart showing distribution of error types."""
    perceptual = sum(1 for p in pairs if p["gap"] > 15)
    reasoning = sum(1 for p in pairs if p["txt_acc"] < 50 and p["img_acc"] < 50)
    mixed = len(pairs) - perceptual - reasoning

    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = [perceptual, reasoning, mixed]
    labels = [f"Perceptual\n({perceptual} tasks)", f"Reasoning\n({reasoning} tasks)", f"Mixed/Minimal\n({mixed} tasks)"]
    colors = ["#e74c3c", "#3498db", "#95a5a6"]
    explode = (0.05, 0.05, 0)

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, explode=explode,
                                       autopct="%1.0f%%", startangle=90, textprops={"fontsize": 10})
    for autotext in autotexts:
        autotext.set_fontweight("bold")

    ax.set_title(f"Error Type Distribution Across {len(pairs)} Tasks", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig9_summary_pie.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig9_summary_pie.png")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("Loading results...")
    tasks = load_results()
    img_tasks, txt_tasks = classify_tasks(tasks)
    pairs = get_pairs(img_tasks, txt_tasks)
    print(f"  {len(pairs)} task pairs found")

    print("\nGenerating figures...")
    fig1_gap_chart(pairs)
    fig2_scatter(pairs)
    fig3_category_bars(pairs)
    fig4_perception_demand(pairs)
    fig5_worst_blindspots(pairs)
    fig6_reasoning_failures(pairs)
    fig7_classification_heatmap(pairs)
    fig8_failure_examples(tasks)
    fig9_summary_pie(pairs)

    print(f"\nAll figures saved to {FIG_DIR}/")

if __name__ == "__main__":
    main()
