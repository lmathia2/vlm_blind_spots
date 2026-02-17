"""Generate custom report figures (main combined, task gallery, image vs text, appendix failures)."""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image

from analysis import TASK_CLASSIFICATION

RESULTS_FILE = sys.argv[1] if len(sys.argv) > 1 else "results_haiku45/results_filtered.jsonl"
FIG_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("report_haiku45/figures")
DATA_DIR = Path("data")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
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


def split_tasks(tasks):
    text_controls = set()
    for name in tasks:
        base = name.rsplit("_text", 1)[0] if name.endswith("_text") else None
        if base and base in tasks:
            text_controls.add(name)
    img = {t: v for t, v in tasks.items() if t not in text_controls}
    txt = {t: v for t, v in tasks.items() if t in text_controls}
    return img, txt


def get_pairs(img_tasks, txt_tasks):
    pairs = []
    for t in sorted(img_tasks):
        tt = t + "_text"
        if tt in txt_tasks:
            img_acc = img_tasks[t]["correct"] / img_tasks[t]["total"] * 100
            txt_acc = txt_tasks[tt]["correct"] / txt_tasks[tt]["total"] * 100
            pairs.append({
                "task": t,
                "img_acc": img_acc,
                "txt_acc": txt_acc,
                "gap": txt_acc - img_acc,
                "img_n": img_tasks[t]["total"],
                "txt_n": txt_tasks[tt]["total"],
            })
    return pairs


# ── Figure: Main combined (left: image vs text bars, right: top 10 blind spots) ──

def fig_main_combined(pairs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.suptitle("VLM Blind Spots: Claude Haiku 4.5 Evaluation", fontsize=16, fontweight="bold", y=0.98)

    # Left panel: all tasks, image vs text
    pairs_sorted = sorted(pairs, key=lambda p: p["img_acc"])
    tasks = [p["task"].replace("_", " ") for p in pairs_sorted]
    img_accs = [p["img_acc"] for p in pairs_sorted]
    txt_accs = [p["txt_acc"] for p in pairs_sorted]

    y = np.arange(len(tasks))
    h = 0.35
    ax1.barh(y - h/2, img_accs, h, label="Image", color="#e74c3c", alpha=0.85)
    ax1.barh(y + h/2, txt_accs, h, label="Text Control", color="#2ecc71", alpha=0.85)
    ax1.set_yticks(y)
    ax1.set_yticklabels(tasks, fontsize=7.5)
    ax1.set_xlabel("Accuracy (%)")
    ax1.set_title("Image vs Text-Only Control Accuracy", fontsize=12, fontweight="bold")
    ax1.set_xlim(0, 110)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(axis="x", alpha=0.2)

    # Right panel: top 10 blind spots (largest gap)
    worst = sorted(pairs, key=lambda p: p["gap"], reverse=True)[:10]
    x = np.arange(len(worst))
    w = 0.35
    ax2.bar(x - w/2, [p["img_acc"] for p in worst], w, label="Image", color="#e74c3c", alpha=0.85)
    ax2.bar(x + w/2, [p["txt_acc"] for p in worst], w, label="Text Control", color="#2ecc71", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([p["task"].replace("_", "\n") for p in worst], fontsize=7.5, rotation=0, ha="center")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Top 10 Perceptual Blind Spots", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 115)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.2)

    for i, p in enumerate(worst):
        ax2.annotate(f"Δ{p['gap']:+.0f}%", (i, max(p["img_acc"], p["txt_acc"]) + 3),
                    ha="center", fontsize=7.5, fontweight="bold", color="#c0392b")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / "fig_main_combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_main_combined.png")


# ── Figure: Task gallery (all tasks with sample images) ──

def fig_task_gallery(pairs, tasks_data):
    # Sort by image accuracy
    pairs_sorted = sorted(pairs, key=lambda p: p["img_acc"])
    n = len(pairs_sorted)
    cols = 6
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(24, 4 * rows))
    fig.suptitle(f"Task Gallery: All {n} Tasks (sorted by image accuracy)",
                 fontsize=16, fontweight="bold", y=1.01)

    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, p in enumerate(pairs_sorted):
        ax = axes_flat[i]
        task_name = p["task"]

        # Find a sample image
        img_shown = False
        if task_name in tasks_data:
            for r in tasks_data[task_name]["records"][:5]:
                img_path = r.get("image_path", "")
                if img_path and os.path.exists(img_path):
                    try:
                        img = Image.open(img_path)
                        ax.imshow(img)
                        img_shown = True
                        break
                    except Exception:
                        pass

        if not img_shown:
            ax.text(0.5, 0.5, task_name.replace("_", "\n"), ha="center", va="center",
                   fontsize=10, transform=ax.transAxes)
            ax.set_facecolor("#f0f0f0")

        # Color-code by accuracy
        acc = p["img_acc"]
        if acc >= 95:
            color = "#22C55E"
        elif acc >= 80:
            color = "#EAB308"
        elif acc >= 60:
            color = "#F97316"
        else:
            color = "#EF4444"

        ax.set_title(f"{task_name.replace('_', ' ')}\n{acc:.0f}% | {p['txt_acc']:.0f}%",
                    fontsize=8, fontweight="bold", color=color)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_task_gallery.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_task_gallery.png")


# ── Figure: Image vs text side-by-side comparison ──

def fig_image_vs_text(pairs, tasks_data):
    # Pick 6 representative tasks across the accuracy spectrum
    target_tasks = ["counting_grid", "colored_paths", "text_degradation",
                    "strikethrough", "pie_chart", "table_cell_read"]
    available = [p for p in pairs if p["task"] in target_tasks]
    available = sorted(available, key=lambda p: p["img_acc"])[:6]

    if not available:
        print("  ⚠ No tasks available for image_vs_text, skipping")
        return

    n = len(available)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n))
    fig.suptitle("Image vs Text-Only Control: Side-by-Side Comparison",
                 fontsize=14, fontweight="bold", y=1.01)

    for i, p in enumerate(available):
        task_name = p["task"]
        ax_img = axes[i][0]
        ax_txt = axes[i][1]

        # Image panel
        img_shown = False
        if task_name in tasks_data:
            for r in tasks_data[task_name]["records"][:10]:
                img_path = r.get("image_path", "")
                if img_path and os.path.exists(img_path):
                    try:
                        img = Image.open(img_path)
                        ax_img.imshow(img)
                        img_shown = True
                        break
                    except Exception:
                        pass

        if not img_shown:
            ax_img.text(0.5, 0.5, "(no image)", ha="center", va="center",
                       transform=ax_img.transAxes)

        ax_img.set_title(f"{task_name.replace('_', ' ')} — Image ({p['img_acc']:.0f}%)",
                        fontsize=10, fontweight="bold")
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        # Text panel
        text_task = task_name + "_text"
        text_shown = False
        if text_task in tasks_data:
            for r in tasks_data[text_task]["records"][:10]:
                prompt = r.get("prompt", "")
                if prompt:
                    # Show the text control prompt
                    wrapped = prompt[:500] + ("..." if len(prompt) > 500 else "")
                    ax_txt.text(0.05, 0.95, wrapped, ha="left", va="top",
                               transform=ax_txt.transAxes, fontsize=7,
                               fontfamily="monospace", wrap=True)
                    text_shown = True
                    break

        if not text_shown:
            ax_txt.text(0.5, 0.5, "(text control)", ha="center", va="center",
                       transform=ax_txt.transAxes)

        ax_txt.set_title(f"{task_name.replace('_', ' ')} — Text Control ({p['txt_acc']:.0f}%)",
                        fontsize=10, fontweight="bold")
        ax_txt.set_xticks([])
        ax_txt.set_yticks([])
        ax_txt.set_facecolor("#f8f9fa")

        # Color-code border by gap
        gap_color = "#e74c3c" if p["gap"] > 15 else "#95a5a6"
        for ax in [ax_img, ax_txt]:
            for spine in ax.spines.values():
                spine.set_edgecolor(gap_color)
                spine.set_linewidth(2)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_image_vs_text.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_image_vs_text.png")


# ── Figure: Appendix failures (6 examples across worst tasks) ──

def fig_appendix_failures(tasks_data, pairs):
    worst_tasks = ["counting_grid", "colored_paths", "nested_squares",
                   "text_degradation", "strikethrough", "pie_chart"]

    examples = []
    for task_name in worst_tasks:
        if task_name not in tasks_data:
            continue
        wrong = [r for r in tasks_data[task_name]["records"] if not r.get("correct")]
        if wrong:
            r = wrong[0]
            img_path = r.get("image_path", "")
            if img_path and os.path.exists(img_path):
                examples.append((task_name, r, img_path))

    if not examples:
        print("  ⚠ No failure examples found, skipping fig_appendix_failures")
        return

    n = min(len(examples), 6)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.suptitle("Representative Failures from Worst-Performing Tasks",
                 fontsize=14, fontweight="bold", y=1.02)

    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    # Lookup accuracy for title
    acc_lookup = {p["task"]: p["img_acc"] for p in pairs}

    for i, (task_name, record, img_path) in enumerate(examples[:n]):
        ax = axes_flat[i]
        img = Image.open(img_path)
        ax.imshow(img)

        gt = record.get("ground_truth", "?")
        ans = record.get("parsed_answer", "?")
        if isinstance(gt, str) and len(gt) > 35:
            gt = gt[:32] + "..."
        if isinstance(ans, str) and len(ans) > 35:
            ans = ans[:32] + "..."

        acc = acc_lookup.get(task_name, 0)
        color = "#EF4444" if acc < 60 else "#F97316" if acc < 80 else "#EAB308"

        ax.set_title(f"{task_name.replace('_', ' ')} ({acc:.0f}%)",
                    fontsize=10, fontweight="bold", color=color)
        ax.set_xlabel(f"GT: {gt}  |  Model: {ans}", fontsize=8, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_appendix_failures.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ fig_appendix_failures.png")


# ── Figure: Taxonomy (3-column: business domain → perceptual primitive → tasks) ──

def fig_taxonomy(pairs):
    acc_lookup = {p["task"]: p["img_acc"] / 100 for p in pairs}

    def acc_color(acc):
        if acc >= 0.95: return "#22C55E"
        elif acc >= 0.80: return "#EAB308"
        elif acc >= 0.60: return "#F97316"
        else: return "#EF4444"

    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(22, 20))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    col1_x, col1_w = 1, 20
    col2_x, col2_w = 26, 22
    col3_x, col3_w = 53, 46

    ax.text(50, 99, "Task Taxonomy: From Business Problems to Evaluation Tasks",
            ha="center", va="center", fontsize=17, fontweight="bold", color="#0F172A")
    ax.text(col1_x + col1_w/2, 96.8, "Business Domain", ha="center", fontsize=14,
            fontweight="bold", color="#1E293B")
    ax.text(col1_x + col1_w/2, 95.3, "What real-world problem?", ha="center",
            fontsize=9, color="#64748B", style="italic")
    ax.text(col2_x + col2_w/2, 96.8, "Perceptual Primitive", ha="center", fontsize=14,
            fontweight="bold", color="#1E293B")
    ax.text(col2_x + col2_w/2, 95.3, "What visual capability?", ha="center",
            fontsize=9, color="#64748B", style="italic")
    ax.text(col3_x + col3_w/2, 96.8, "Evaluation Tasks", ha="center", fontsize=14,
            fontweight="bold", color="#1E293B")
    ax.text(col3_x + col3_w/2, 95.3, "How do we test it? (with image accuracy)", ha="center",
            fontsize=9, color="#64748B", style="italic")

    domains = [
        ("Tabular Data &\nSpreadsheets", "PDFs, invoices,\nfinancial reports", "#3B82F6"),
        ("Charts & Data\nVisualization", "Dashboards, KPIs,\ntrend analysis", "#10B981"),
        ("Diagrams &\nFlowcharts", "Process flows, org charts,\nnetwork diagrams", "#F59E0B"),
        ("Document Layout\n& Forms", "Intake forms, contracts,\nfield extraction", "#8B5CF6"),
        ("Annotations &\nMarkup", "Reviews, corrections,\ntracked changes", "#EC4899"),
        ("Dashboards\n& UI", "Progress indicators,\nstatus monitoring", "#6366F1"),
    ]

    dom_h = 10.5
    total_dom = len(domains) * dom_h
    dom_gap = (92 - total_dom) / (len(domains) + 1)
    dom_centers = {}

    for i, (name, desc, color) in enumerate(domains):
        y = 92 - (i + 1) * dom_gap - i * dom_h - dom_h
        box = FancyBboxPatch((col1_x, y), col1_w, dom_h,
                              boxstyle="round,pad=0.4",
                              facecolor=color, edgecolor="none", alpha=0.92, zorder=3)
        ax.add_patch(box)
        ax.text(col1_x + col1_w/2, y + dom_h * 0.62, name, ha="center", va="center",
                fontsize=10.5, color="white", fontweight="bold", zorder=4)
        ax.text(col1_x + col1_w/2, y + dom_h * 0.22, desc, ha="center", va="center",
                fontsize=7.5, color="white", alpha=0.85, zorder=4)
        dom_centers[i] = (col1_x + col1_w, y + dom_h/2, color)

    primitives = [
        ("P1  Spatial\nReference", "Locate row 3 col 5,\ncontainment, adjacency"),
        ("P2  Line / Path\nFollowing", "Trace paths through\ncrossings and turns"),
        ("P3  Counting &\nEnumeration", "Count objects, rows,\nnesting levels"),
        ("P4  Intersection\nDetection", "Determine if/where\nlines cross"),
        ("P5  Fine State\nDiscrimination", "Checked vs unchecked,\ntouching vs separated"),
        ("P7  Color\nMapping", "Match colors to legends,\ndistinguish hues"),
        ("P8  Text in Visual\nContext", "Read text at varied sizes,\nrotation, degradation"),
        ("P9  Scale &\nProportion", "Map visual magnitude\nto numeric value"),
    ]

    prim_to_tasks = {
        0: [("table_cell_read", acc_lookup.get("table_cell_read", 0)),
            ("realistic_table", acc_lookup.get("realistic_table", 0)),
            ("merged_cell_read", acc_lookup.get("merged_cell_read", 0)),
            ("form_field", acc_lookup.get("form_field", 0))],
        1: [("arrow_following", acc_lookup.get("arrow_following", 0)),
            ("decision_flowchart", acc_lookup.get("decision_flowchart", 0)),
            ("colored_paths", acc_lookup.get("colored_paths", 0)),
            ("hierarchy_depth", acc_lookup.get("hierarchy_depth", 0))],
        2: [("counting_grid", acc_lookup.get("counting_grid", 0)),
            ("nested_squares", acc_lookup.get("nested_squares", 0)),
            ("line_chart_crossing", acc_lookup.get("line_chart_crossing", 0))],
        3: [("edge_crossing", acc_lookup.get("edge_crossing", 0))],
        4: [("form_checkboxes", acc_lookup.get("form_checkboxes", 0)),
            ("radio_button", acc_lookup.get("radio_button", 0)),
            ("touching_circles", acc_lookup.get("touching_circles", 0)),
            ("strikethrough", acc_lookup.get("strikethrough", 0))],
        5: [("legend_association", acc_lookup.get("legend_association", 0)),
            ("color_coded_cells", acc_lookup.get("color_coded_cells", 0)),
            ("heatmap", acc_lookup.get("heatmap", 0))],
        6: [("text_degradation", acc_lookup.get("text_degradation", 0)),
            ("rotated_text", acc_lookup.get("rotated_text", 0)),
            ("dense_text", acc_lookup.get("dense_text", 0)),
            ("circled_text", acc_lookup.get("circled_text", 0)),
            ("highlighted_text", acc_lookup.get("highlighted_text", 0)),
            ("arrow_annotation", acc_lookup.get("arrow_annotation", 0))],
        7: [("bar_chart_value", acc_lookup.get("bar_chart_value", 0)),
            ("pie_chart", acc_lookup.get("pie_chart", 0)),
            ("scatter_plot", acc_lookup.get("scatter_plot", 0)),
            ("grouped_bar", acc_lookup.get("grouped_bar", 0)),
            ("stacked_bar", acc_lookup.get("stacked_bar", 0)),
            ("line_chart_point", acc_lookup.get("line_chart_point", 0)),
            ("progress_bar", acc_lookup.get("progress_bar", 0))],
    }

    task_h = 2.5
    task_gap = 0.25
    prim_internal_pad = 1.0
    group_gap = 1.2

    group_heights = []
    for pi in range(len(primitives)):
        tasks = prim_to_tasks.get(pi, [])
        n = max(len(tasks), 1)
        h = n * task_h + (n - 1) * task_gap + 2 * prim_internal_pad
        group_heights.append(h)

    total_height = sum(group_heights) + (len(primitives) - 1) * group_gap
    start_y = 93
    scale = min(1.0, (start_y - 3) / total_height)

    current_y = start_y
    prim_centers = {}

    for pi in range(len(primitives)):
        tasks = prim_to_tasks.get(pi, [])
        gh = group_heights[pi] * scale
        actual_task_h = task_h * scale
        actual_task_gap = task_gap * scale
        actual_pad = prim_internal_pad * scale

        group_top = current_y
        group_bot = current_y - gh

        prim_box_h = min(gh, 8)
        prim_y = group_bot + (gh - prim_box_h) / 2
        name, desc = primitives[pi]

        box = FancyBboxPatch((col2_x, prim_y), col2_w, prim_box_h,
                              boxstyle="round,pad=0.3",
                              facecolor="#F8FAFC", edgecolor="#CBD5E1",
                              linewidth=1.5, zorder=3)
        ax.add_patch(box)
        ax.text(col2_x + col2_w/2, prim_y + prim_box_h * 0.62, name,
                ha="center", va="center", fontsize=9, color="#1E293B",
                fontweight="bold", zorder=4)
        ax.text(col2_x + col2_w/2, prim_y + prim_box_h * 0.25, desc,
                ha="center", va="center", fontsize=7, color="#64748B", zorder=4)

        prim_cy = prim_y + prim_box_h / 2
        prim_centers[pi] = (col2_x, prim_cy, col2_x + col2_w, prim_cy)

        for j, (tname, acc) in enumerate(tasks):
            ty = group_top - actual_pad - j * (actual_task_h + actual_task_gap) - actual_task_h
            color = acc_color(acc)
            display = tname.replace("_", " ")

            pill = FancyBboxPatch((col3_x, ty), col3_w, actual_task_h,
                                   boxstyle="round,pad=0.15",
                                   facecolor="white", edgecolor=color,
                                   linewidth=2.2, zorder=3)
            ax.add_patch(pill)

            badge_w = 5.5
            badge = FancyBboxPatch((col3_x + col3_w - badge_w - 0.4, ty + 0.2),
                                    badge_w, actual_task_h - 0.4,
                                    boxstyle="round,pad=0.12",
                                    facecolor=color, edgecolor="none", zorder=4)
            ax.add_patch(badge)
            ax.text(col3_x + col3_w - badge_w/2 - 0.4, ty + actual_task_h/2,
                    f"{acc:.0%}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold", zorder=5)

            ax.text(col3_x + 1.2, ty + actual_task_h/2, display,
                    ha="left", va="center", fontsize=8.5, color="#374151", zorder=4)

        if tasks:
            tasks_mid_y = group_top - actual_pad - (len(tasks) * (actual_task_h + actual_task_gap) - actual_task_gap) / 2
            ax.annotate("", xy=(col3_x - 0.3, tasks_mid_y),
                        xytext=(col2_x + col2_w + 0.3, prim_cy),
                        arrowprops=dict(arrowstyle="-|>", color="#94A3B8",
                                       alpha=0.5, lw=1.3,
                                       connectionstyle="arc3,rad=0.03"))

        current_y = group_bot - group_gap * scale

    dom_to_prim = {
        0: [0, 2, 6],
        1: [3, 5, 7],
        2: [1, 2, 4],
        3: [0, 4, 6],
        4: [0, 5, 6],
        5: [4, 7],
    }

    for dom_idx, prim_indices in dom_to_prim.items():
        dx, dy, dcolor = dom_centers[dom_idx]
        for pi in prim_indices:
            px_l, py, _, _ = prim_centers[pi]
            ax.annotate("", xy=(px_l - 0.3, py),
                        xytext=(dx + 0.3, dy),
                        arrowprops=dict(arrowstyle="-", color=dcolor,
                                       alpha=0.22, lw=1.5,
                                       connectionstyle="arc3,rad=0.08"))

    legend_y = 0.5
    ax.text(col3_x, legend_y + 1.2, "Image Accuracy:", fontsize=9.5,
            fontweight="bold", color="#475569")
    items = [("\u2265 95%", "#22C55E"), ("80\u201394%", "#EAB308"),
             ("60\u201379%", "#F97316"), ("< 60%", "#EF4444")]
    for i, (label, color) in enumerate(items):
        lx = col3_x + 14 + i * 9
        b = FancyBboxPatch((lx, legend_y + 0.2), 5.5, 2.2,
                            boxstyle="round,pad=0.12",
                            facecolor=color, edgecolor="none", zorder=4)
        ax.add_patch(b)
        ax.text(lx + 2.75, legend_y + 1.3, label, ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold", zorder=5)

    fig.tight_layout(pad=0.5)
    fig.savefig(FIG_DIR / "fig_taxonomy.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ✓ fig_taxonomy.png")


def main():
    print("Loading results...")
    tasks_data = load_data()
    img_tasks, txt_tasks = split_tasks(tasks_data)
    pairs = get_pairs(img_tasks, txt_tasks)
    print(f"  {len(pairs)} task pairs found")

    print("\nGenerating custom figures...")
    fig_main_combined(pairs)
    fig_task_gallery(pairs, tasks_data)
    fig_image_vs_text(pairs, tasks_data)
    fig_appendix_failures(tasks_data, pairs)
    fig_taxonomy(pairs)
    print(f"\nAll custom figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
