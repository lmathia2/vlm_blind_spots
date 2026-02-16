"""Analysis: summary tables, accuracy curves, confusion matrices, failure examples."""

import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd


def load_results(results_path: str | Path) -> list[dict]:
    """Load results JSONL into a list of dicts."""
    results = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def wilson_ci(n_correct: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion.

    Returns (lower, upper) bounds as fractions in [0, 1].
    """
    if n_total == 0:
        return (0.0, 0.0)
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    half_width = (z / denom) * math.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2))
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def print_summary(results_path: str | Path):
    """Print accuracy summary table grouped by task_name.

    If results contain both reasoning_mode=True and False, prints a comparison table.
    """
    results = load_results(results_path)
    if not results:
        print("No results found.")
        return

    df = pd.DataFrame(results)

    # Normalize: missing reasoning_mode → False
    if "reasoning_mode" not in df.columns:
        df["reasoning_mode"] = False
    else:
        df.loc[df["reasoning_mode"].isna(), "reasoning_mode"] = False
        df["reasoning_mode"] = df["reasoning_mode"].astype(bool)

    modes = df["reasoning_mode"].unique()
    has_both = True in modes and False in modes

    if has_both:
        _print_comparison_summary(df)
    else:
        _print_single_summary(df)


def _print_single_summary(df: pd.DataFrame):
    """Print single-mode accuracy summary."""
    grouped = df.groupby("task_name")
    mode_label = ""
    if "reasoning_mode" in df.columns and df["reasoning_mode"].iloc[0]:
        mode_label = " [REASONING]"

    print(f"\n{'Task':<25} {'N':>5} {'Correct':>8} {'Accuracy':>9} {'95% CI':>17} {'Parse Fail':>11}{mode_label}")
    print("-" * 79)

    for task, group in sorted(grouped):
        n = len(group)
        correct = int(group["correct"].sum())
        accuracy = correct / n if n > 0 else 0
        lo, hi = wilson_ci(correct, n)
        parse_fails = group["parsed_answer"].isna().sum() if "parsed_answer" in group.columns else 0
        pf_rate = parse_fails / n if n > 0 else 0

        print(f"{task:<25} {n:>5} {correct:>8} {accuracy:>8.1%} [{lo:>5.1%}, {hi:>5.1%}] {pf_rate:>10.1%}")

        if "error" in group.columns and group["error"].notna().any():
            mean_err = group["error"].dropna().mean()
            print(f"  {'':>25} mean_error={mean_err:+.2f}")

    total_n = len(df)
    total_correct = int(df["correct"].sum())
    total_acc = df["correct"].mean()
    lo, hi = wilson_ci(total_correct, total_n)
    print(f"\n{'TOTAL':<25} {total_n:>5} {total_correct:>8} "
          f"{total_acc:>8.1%} [{lo:>5.1%}, {hi:>5.1%}]")


def _print_comparison_summary(df: pd.DataFrame):
    """Print side-by-side comparison of reasoning vs no-reasoning."""
    df_no = df[df["reasoning_mode"] == False]
    df_yes = df[df["reasoning_mode"] == True]

    all_tasks = sorted(set(df["task_name"]))

    print(f"\n{'Task':<25} {'N':>4} {'No-Reason':>13} {'N':>4} {'Reasoning':>13} {'Delta':>8}")
    print("-" * 73)

    total_no_n, total_no_c = 0, 0
    total_yes_n, total_yes_c = 0, 0

    for task in all_tasks:
        no_group = df_no[df_no["task_name"] == task]
        yes_group = df_yes[df_yes["task_name"] == task]

        n_no = len(no_group)
        c_no = int(no_group["correct"].sum()) if n_no > 0 else 0
        acc_no = c_no / n_no if n_no > 0 else 0

        n_yes = len(yes_group)
        c_yes = int(yes_group["correct"].sum()) if n_yes > 0 else 0
        acc_yes = c_yes / n_yes if n_yes > 0 else 0

        total_no_n += n_no
        total_no_c += c_no
        total_yes_n += n_yes
        total_yes_c += c_yes

        delta = acc_yes - acc_no if n_no > 0 and n_yes > 0 else 0
        delta_str = f"{delta:+.1%}" if n_no > 0 and n_yes > 0 else "—"

        no_str = f"{acc_no:.1%}" if n_no > 0 else "—"
        yes_str = f"{acc_yes:.1%}" if n_yes > 0 else "—"

        # Color delta
        if delta > 0.005:
            delta_str = f"\033[32m{delta_str}\033[0m"
        elif delta < -0.005:
            delta_str = f"\033[31m{delta_str}\033[0m"

        print(f"{task:<25} {n_no:>4} {no_str:>13} {n_yes:>4} {yes_str:>13} {delta_str:>8}")

    # Totals
    acc_no_total = total_no_c / total_no_n if total_no_n > 0 else 0
    acc_yes_total = total_yes_c / total_yes_n if total_yes_n > 0 else 0
    delta_total = acc_yes_total - acc_no_total
    print(f"\n{'TOTAL':<25} {total_no_n:>4} {acc_no_total:>12.1%} {total_yes_n:>4} {acc_yes_total:>12.1%} {delta_total:>+7.1%}")


def generate_all_plots(results_path: str | Path):
    """Generate all analysis plots for a results file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    from config import REPORT_ASSETS_DIR

    results = load_results(results_path)
    if not results:
        return

    df = pd.DataFrame(results)

    # Normalize reasoning_mode for older results
    if "reasoning_mode" not in df.columns:
        df["reasoning_mode"] = False
    else:
        df.loc[df["reasoning_mode"].isna(), "reasoning_mode"] = False
        df["reasoning_mode"] = df["reasoning_mode"].astype(bool)

    output_dir = REPORT_ASSETS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Accuracy by task bar chart with CI error bars
    task_stats = df.groupby("task_name")["correct"].agg(["sum", "count"])
    task_stats["accuracy"] = task_stats["sum"] / task_stats["count"]
    task_stats[["ci_lo", "ci_hi"]] = task_stats.apply(
        lambda r: pd.Series(wilson_ci(int(r["sum"]), int(r["count"]))), axis=1
    )
    task_stats = task_stats.sort_values("accuracy")

    fig, ax = plt.subplots(figsize=(10, max(4, len(task_stats) * 0.5)))
    y_pos = range(len(task_stats))
    xerr_lo = (task_stats["accuracy"] - task_stats["ci_lo"]).clip(lower=0)
    xerr_hi = (task_stats["ci_hi"] - task_stats["accuracy"]).clip(lower=0)
    ax.barh(y_pos, task_stats["accuracy"], color="steelblue", xerr=[xerr_lo, xerr_hi],
            capsize=3, ecolor="gray")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(task_stats.index)
    ax.set_xlabel("Accuracy")
    ax.set_title("Accuracy by Task (95% Wilson CI)")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_by_task.png", dpi=150)
    plt.close(fig)
    print(f"Saved accuracy_by_task.png")

    # Per-task plots
    for task_name, group in df.groupby("task_name"):
        _plot_task_details(group, task_name, output_dir)

    # Reasoning comparison plot (if both modes present)
    if "reasoning_mode" in df.columns:
        modes = df["reasoning_mode"].unique()
        if True in modes and False in modes:
            _plot_reasoning_comparison(df, output_dir)

    # Perception vs reasoning plot (if text-only controls exist)
    pairs = _discover_perception_pairs(df["task_name"].unique().tolist())
    if pairs:
        _plot_perception_vs_reasoning(df, pairs, output_dir)


def _plot_reasoning_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot grouped bar chart comparing reasoning vs no-reasoning accuracy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    df_no = df[df["reasoning_mode"] == False]
    df_yes = df[df["reasoning_mode"] == True]

    tasks = sorted(set(df["task_name"]))
    acc_no = [df_no[df_no["task_name"] == t]["correct"].mean() if len(df_no[df_no["task_name"] == t]) > 0 else 0 for t in tasks]
    acc_yes = [df_yes[df_yes["task_name"] == t]["correct"].mean() if len(df_yes[df_yes["task_name"] == t]) > 0 else 0 for t in tasks]

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, max(5, len(tasks) * 0.6)))
    bars1 = ax.barh(x - width / 2, acc_no, width, label="No Reasoning", color="steelblue")
    bars2 = ax.barh(x + width / 2, acc_yes, width, label="Reasoning", color="coral")

    ax.set_yticks(x)
    ax.set_yticklabels(tasks)
    ax.set_xlabel("Accuracy")
    ax.set_title("Accuracy: Reasoning vs No Reasoning")
    ax.set_xlim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "reasoning_comparison.png", dpi=150)
    plt.close(fig)
    print("Saved reasoning_comparison.png")


def _plot_task_details(group: pd.DataFrame, task_name: str, output_dir: Path):
    """Generate detailed plots for a single task."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # If we have params, try accuracy vs each param
    if "params" in group.columns and group["params"].notna().any():
        params_df = pd.json_normalize(group["params"])
        for col in params_df.columns:
            # Skip non-scalar columns (lists, dicts) and string columns
            if params_df[col].apply(lambda x: isinstance(x, (list, dict, str))).any():
                continue
            try:
                if params_df[col].nunique() <= 1:
                    continue
            except TypeError:
                continue
            if params_df[col].nunique() > 1:
                merged = pd.concat([group.reset_index(drop=True), params_df], axis=1)
                acc_by_param = merged.groupby(col)["correct"].mean()
                if len(acc_by_param) > 1:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    acc_by_param.plot(kind="bar", ax=ax, color="steelblue")
                    ax.set_ylabel("Accuracy")
                    ax.set_title(f"{task_name}: Accuracy vs {col}")
                    ax.set_ylim(0, 1)
                    fig.tight_layout()
                    fig.savefig(output_dir / f"{task_name}_acc_vs_{col}.png", dpi=150)
                    plt.close(fig)

    # Confusion matrix for counting tasks (integer predictions)
    if "error" in group.columns and group["error"].notna().any():
        _plot_confusion(group, task_name, output_dir)


def _plot_confusion(group: pd.DataFrame, task_name: str, output_dir: Path):
    """Plot predicted vs ground truth confusion heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    try:
        gt_vals = group["ground_truth"].astype(int)
        pred_vals = group["parsed_answer"].dropna().astype(int)
    except (ValueError, TypeError):
        return

    common_idx = gt_vals.index.intersection(pred_vals.index)
    if len(common_idx) < 5:
        return

    gt = gt_vals.loc[common_idx]
    pred = pred_vals.loc[common_idx]
    all_vals = sorted(set(gt) | set(pred))

    confusion = pd.crosstab(pred, gt, dropna=False).reindex(
        index=all_vals, columns=all_vals, fill_value=0
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{task_name}: Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / f"{task_name}_confusion.png", dpi=150)
    plt.close(fig)


def compute_bias(results_path: str | Path, task_name: str) -> dict:
    """Compute over/undercounting bias for a counting task."""
    results = load_results(results_path)
    df = pd.DataFrame(results)
    task_df = df[df["task_name"] == task_name]

    if "error" not in task_df.columns:
        return {}

    errors = task_df["error"].dropna()
    if len(errors) == 0:
        return {}

    return {
        "mean_error": float(errors.mean()),
        "overcount_rate": float((errors > 0).mean()),
        "undercount_rate": float((errors < 0).mean()),
        "exact_rate": float((errors == 0).mean()),
    }


CLUTTER_TAX_PAIRS = [
    ("line_intersection", "line_chart_crossing", "P4 Intersection Detection"),
    ("touching_circles", "form_checkboxes", "P5 Fine State Discrimination"),
    ("colored_paths", "arrow_following", "P2 Path Following"),
]


def print_clutter_tax(results_path: str | Path):
    """Print matched-pair comparison: clean geometric vs business-context accuracy."""
    results = load_results(results_path)
    if not results:
        print("No results found.")
        return

    df = pd.DataFrame(results)
    # Use non-reasoning results only
    if "reasoning_mode" in df.columns:
        df = df[(df["reasoning_mode"].isna()) | (df["reasoning_mode"] == False)]

    print(f"\n{'Clean (Geometric)':<25} {'Business Context':<25} {'Primitive':<30} {'Clean':>7} {'Biz':>7} {'Gap':>7}")
    print("-" * 108)

    for clean_task, biz_task, primitive in CLUTTER_TAX_PAIRS:
        clean_df = df[df["task_name"] == clean_task]
        biz_df = df[df["task_name"] == biz_task]

        clean_acc = clean_df["correct"].mean() if len(clean_df) > 0 else None
        biz_acc = biz_df["correct"].mean() if len(biz_df) > 0 else None

        clean_str = f"{clean_acc:.1%}" if clean_acc is not None else "—"
        biz_str = f"{biz_acc:.1%}" if biz_acc is not None else "—"

        if clean_acc is not None and biz_acc is not None:
            gap = biz_acc - clean_acc
            gap_str = f"{gap:+.1%}"
        else:
            gap_str = "—"

        print(f"{clean_task:<25} {biz_task:<25} {primitive:<30} {clean_str:>7} {biz_str:>7} {gap_str:>7}")


def plot_clutter_tax(results_path: str | Path):
    """Generate grouped bar chart comparing clean vs business-context accuracy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from config import REPORT_ASSETS_DIR

    results = load_results(results_path)
    if not results:
        return

    df = pd.DataFrame(results)
    if "reasoning_mode" in df.columns:
        df = df[(df["reasoning_mode"].isna()) | (df["reasoning_mode"] == False)]

    output_dir = REPORT_ASSETS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    clean_accs = []
    biz_accs = []

    for clean_task, biz_task, primitive in CLUTTER_TAX_PAIRS:
        clean_df = df[df["task_name"] == clean_task]
        biz_df = df[df["task_name"] == biz_task]
        if len(clean_df) == 0 and len(biz_df) == 0:
            continue
        labels.append(primitive.split(" ", 1)[1] if " " in primitive else primitive)
        clean_accs.append(clean_df["correct"].mean() if len(clean_df) > 0 else 0)
        biz_accs.append(biz_df["correct"].mean() if len(biz_df) > 0 else 0)

    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, clean_accs, width, label="Clean (Geometric)", color="steelblue")
    ax.bar(x + width / 2, biz_accs, width, label="Business Context", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Clutter Tax: Clean vs Business Context")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "clutter_tax.png", dpi=150)
    plt.close(fig)
    print("Saved clutter_tax.png")


def save_failure_examples(results_path: str | Path, output_dir: str | Path, n: int = 20):
    """Copy the N worst failures to output directory."""
    import shutil

    results = load_results(results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort by score ascending (worst first)
    failures = [r for r in results if not r.get("correct", True)]
    failures.sort(key=lambda r: r.get("score", 0))

    for i, fail in enumerate(failures[:n]):
        img_path = Path(fail.get("image_path", ""))
        if img_path.exists():
            dest = output_dir / f"fail_{i:03d}_{fail.get('task_name', 'unknown')}_{img_path.name}"
            shutil.copy2(img_path, dest)

    # Write metadata
    meta_path = output_dir / "failures.jsonl"
    with open(meta_path, "w") as f:
        for fail in failures[:n]:
            f.write(json.dumps(fail) + "\n")

    print(f"Saved {min(n, len(failures))} failure examples to {output_dir}")


# ---------- Task Classification Taxonomy ----------

TASK_CLASSIFICATION = {
    # Pure perception (just see and read)
    "rotated_text":       {"perception": "high",   "reasoning": "none",   "category": "text_reading"},
    "text_degradation":   {"perception": "high",   "reasoning": "none",   "category": "text_reading"},
    "dense_text":         {"perception": "high",   "reasoning": "low",    "category": "text_reading"},
    "circled_text":       {"perception": "medium", "reasoning": "none",   "category": "annotation_detection"},
    "strikethrough":      {"perception": "medium", "reasoning": "none",   "category": "annotation_detection"},
    "highlighted_text":   {"perception": "medium", "reasoning": "none",   "category": "annotation_detection"},
    "form_checkboxes":    {"perception": "medium", "reasoning": "none",   "category": "ui_state_reading"},
    "radio_button":       {"perception": "medium", "reasoning": "none",   "category": "ui_state_reading"},
    "touching_circles":   {"perception": "high",   "reasoning": "low",    "category": "spatial_discrimination"},

    # Perception + lookup (see and retrieve)
    "table_cell_read":    {"perception": "medium", "reasoning": "low",    "category": "table_lookup"},
    "form_field":         {"perception": "medium", "reasoning": "low",    "category": "table_lookup"},
    "realistic_table":    {"perception": "medium", "reasoning": "low",    "category": "table_lookup"},
    "merged_cell_read":   {"perception": "medium", "reasoning": "medium", "category": "table_lookup"},
    "arrow_annotation":   {"perception": "medium", "reasoning": "low",    "category": "annotation_detection"},
    "color_coded_cells":  {"perception": "medium", "reasoning": "low",    "category": "color_discrimination"},

    # Scale reading (read values from visual encodings)
    "bar_chart_value":    {"perception": "high",   "reasoning": "low",    "category": "scale_reading"},
    "scatter_plot":       {"perception": "high",   "reasoning": "low",    "category": "scale_reading"},
    "heatmap":            {"perception": "high",   "reasoning": "low",    "category": "scale_reading"},
    "grouped_bar":        {"perception": "high",   "reasoning": "medium", "category": "scale_reading"},
    "stacked_bar":        {"perception": "high",   "reasoning": "medium", "category": "scale_reading"},
    "line_chart_point":   {"perception": "high",   "reasoning": "low",    "category": "scale_reading"},
    "progress_bar":       {"perception": "medium", "reasoning": "low",    "category": "scale_reading"},
    "pie_chart":          {"perception": "high",   "reasoning": "low",    "category": "scale_reading"},
    "legend_association": {"perception": "medium", "reasoning": "medium", "category": "chart_comprehension"},
    "line_style":         {"perception": "high",   "reasoning": "none",   "category": "style_discrimination"},

    # Perception + reasoning (see, then compute/follow/count)
    "counting_grid":      {"perception": "high",   "reasoning": "medium", "category": "counting"},
    "nested_squares":     {"perception": "high",   "reasoning": "low",    "category": "counting"},
    "line_intersection":  {"perception": "high",   "reasoning": "high",   "category": "counting"},
    "line_chart_crossing": {"perception": "high",  "reasoning": "medium", "category": "counting"},
    "decision_flowchart": {"perception": "medium", "reasoning": "high",   "category": "graph_traversal"},
    "arrow_following":    {"perception": "medium", "reasoning": "medium", "category": "graph_traversal"},
    "colored_paths":      {"perception": "high",   "reasoning": "medium", "category": "path_following"},
    "edge_crossing":      {"perception": "high",   "reasoning": "medium", "category": "graph_analysis"},
    "hierarchy_depth":    {"perception": "medium", "reasoning": "medium", "category": "graph_analysis"},
    "venn_diagram":       {"perception": "medium", "reasoning": "high",   "category": "set_reasoning"},
}


# ---------- Perception vs Reasoning Diagnostic ----------

def _discover_perception_pairs(task_names: list[str]) -> list[tuple[str, str]]:
    """Auto-discover image/text control pairs from task names.

    A pair exists when both 'foo' and 'foo_text' are registered.
    """
    name_set = set(task_names)
    pairs = []
    for name in sorted(task_names):
        text_name = f"{name}_text"
        if text_name in name_set:
            pairs.append((name, text_name))
    return pairs


def _classify_failure(image_acc: float | None, text_acc: float | None) -> str:
    """Classify failure mode based on image vs text accuracy gap.

    Returns one of:
      - "perceptual" — text >> image: model reasons fine but can't see it
      - "reasoning" — text ≈ image (both low): model can't solve it even with text
      - "mixed" — text > image but both poor: partial perceptual, partial reasoning
      - "not_a_failure" — image accuracy is already high (>80%)
      - "insufficient_data" — one or both missing
    """
    if image_acc is None or text_acc is None:
        return "insufficient_data"
    if image_acc > 0.80:
        return "not_a_failure"
    gap = text_acc - image_acc
    if text_acc > 0.80 and gap > 0.15:
        return "perceptual"
    if gap < 0.10 and text_acc < 0.80:
        return "reasoning"
    if gap >= 0.10:
        return "mixed"
    return "reasoning"


def _plot_perception_vs_reasoning(df: pd.DataFrame, pairs: list[tuple[str, str]], output_dir: Path):
    """Generate grouped bar chart comparing image vs text-only accuracy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = []
    img_accs = []
    txt_accs = []

    for image_task, text_task in pairs:
        img_df = df[df["task_name"] == image_task]
        txt_df = df[df["task_name"] == text_task]
        if len(img_df) == 0 and len(txt_df) == 0:
            continue
        labels.append(image_task.replace("_", " ").title())
        img_accs.append(img_df["correct"].mean() if len(img_df) > 0 else 0)
        txt_accs.append(txt_df["correct"].mean() if len(txt_df) > 0 else 0)

    if not labels:
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.6), 5))
    ax.bar(x - width / 2, img_accs, width, label="Image-based", color="steelblue")
    ax.bar(x + width / 2, txt_accs, width, label="Text-only", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Accuracy")
    ax.set_title("Perception vs Reasoning Diagnostic")
    ax.set_ylim(0, 1.15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "perception_vs_reasoning.png", dpi=150)
    plt.close(fig)
    print("Saved perception_vs_reasoning.png")


def print_full_diagnostic(results_path: str | Path):
    """Print comprehensive diagnostic table for all tasks with text control pairs.

    Shows image accuracy, text accuracy, gap, empirical classification,
    theoretical perception/reasoning demands, and category for every task.
    Also prints summaries by category and error type.
    """
    results = load_results(results_path)
    if not results:
        print("No results found.")
        return

    df = pd.DataFrame(results)

    # Normalize reasoning_mode
    if "reasoning_mode" in df.columns:
        df.loc[df["reasoning_mode"].isna(), "reasoning_mode"] = False
        df["reasoning_mode"] = df["reasoning_mode"].astype(bool)
    else:
        df["reasoning_mode"] = False

    # Use all results (reasoning mode is now default)
    all_task_names = sorted(df["task_name"].unique())
    pairs = _discover_perception_pairs(all_task_names)

    # Build pair lookup: image_task -> text_task
    text_for = {img: txt for img, txt in pairs}

    # Image tasks are those that appear as the first element of a pair,
    # plus any tasks that don't have a pair at all and aren't text controls
    text_control_set = {txt for _, txt in pairs}
    image_tasks = sorted(t for t in all_task_names if t not in text_control_set)

    # Color codes
    class_colors = {
        "perceptual": "\033[33m",
        "reasoning": "\033[31m",
        "mixed": "\033[35m",
        "not_a_failure": "\033[32m",
        "insufficient_data": "\033[90m",
    }
    reset = "\033[0m"

    demand_colors = {
        "none": "\033[90m",
        "low": "\033[32m",
        "medium": "\033[33m",
        "high": "\033[31m",
    }

    # Header
    print("\n┌─ Full Perception vs Reasoning Diagnostic ─────────────────────────────────────────────────────────────────────┐")
    print(f"  {'Task':<24} {'Img Acc':>8} {'Txt Acc':>8} {'Gap':>8} {'Empirical':<16} {'P-demand':>9} {'R-demand':>9} {'Category':<22}")
    print("  " + "─" * 108)

    rows_data = []
    for task in image_tasks:
        img_df = df[df["task_name"] == task]
        img_acc = img_df["correct"].mean() if len(img_df) > 0 else None

        text_task = text_for.get(task)
        txt_acc = None
        if text_task:
            txt_df = df[df["task_name"] == text_task]
            txt_acc = txt_df["correct"].mean() if len(txt_df) > 0 else None

        classification = _classify_failure(img_acc, txt_acc)
        tc = TASK_CLASSIFICATION.get(task, {})
        p_demand = tc.get("perception", "—")
        r_demand = tc.get("reasoning", "—")
        category = tc.get("category", "—")

        img_str = f"{img_acc:.1%}" if img_acc is not None else "—"
        txt_str = f"{txt_acc:.1%}" if txt_acc is not None else "—"
        if img_acc is not None and txt_acc is not None:
            gap = txt_acc - img_acc
            gap_str = f"{gap:+.1%}"
        else:
            gap_str = "—"

        cc = class_colors.get(classification, "")
        pc = demand_colors.get(p_demand, "")
        rc = demand_colors.get(r_demand, "")

        print(
            f"  {task:<24} {img_str:>8} {txt_str:>8} {gap_str:>8} "
            f"{cc}{classification:<16}{reset} "
            f"{pc}{p_demand:>9}{reset} {rc}{r_demand:>9}{reset} {category:<22}"
        )

        rows_data.append({
            "task": task,
            "img_acc": img_acc,
            "txt_acc": txt_acc,
            "classification": classification,
            "p_demand": p_demand,
            "r_demand": r_demand,
            "category": category,
        })

    print("└" + "─" * 112 + "┘")

    # --- Summary by error type ---
    print("\n┌─ Worst Tasks by Error Type ───────────────────────────────────────┐")

    # Perceptual failures (largest positive gap where text >> image)
    perceptual = [
        r for r in rows_data
        if r["img_acc"] is not None and r["txt_acc"] is not None
        and r["classification"] == "perceptual"
    ]
    perceptual.sort(key=lambda r: (r["txt_acc"] - r["img_acc"]), reverse=True)

    print(f"\n  \033[33mPerceptual bottlenecks\033[0m (text >> image, model can't see it):")
    if perceptual:
        for r in perceptual[:10]:
            gap = r["txt_acc"] - r["img_acc"]
            print(f"    {r['task']:<24} img={r['img_acc']:.1%}  txt={r['txt_acc']:.1%}  gap={gap:+.1%}")
    else:
        print("    (none found)")

    # Reasoning failures (both text and image low)
    reasoning = [
        r for r in rows_data
        if r["img_acc"] is not None and r["txt_acc"] is not None
        and r["classification"] == "reasoning"
    ]
    reasoning.sort(key=lambda r: r["txt_acc"])

    print(f"\n  \033[31mReasoning bottlenecks\033[0m (both text and image low):")
    if reasoning:
        for r in reasoning[:10]:
            print(f"    {r['task']:<24} img={r['img_acc']:.1%}  txt={r['txt_acc']:.1%}")
    else:
        print("    (none found)")

    # Mixed
    mixed = [
        r for r in rows_data
        if r["classification"] == "mixed"
    ]
    mixed.sort(key=lambda r: r["img_acc"] or 0)

    print(f"\n  \033[35mMixed failures\033[0m (text > image but both poor):")
    if mixed:
        for r in mixed[:10]:
            gap = (r["txt_acc"] or 0) - (r["img_acc"] or 0)
            print(f"    {r['task']:<24} img={r['img_acc']:.1%}  txt={r['txt_acc']:.1%}  gap={gap:+.1%}")
    else:
        print("    (none found)")

    print("\n└" + "─" * 67 + "┘")

    # --- Summary by category ---
    print("\n┌─ Summary by Category ─────────────────────────────────────────────┐")
    print(f"  {'Category':<22} {'N Tasks':>8} {'Mean Img':>9} {'Mean Txt':>9} {'Mean Gap':>9}")
    print("  " + "─" * 60)

    categories = defaultdict(lambda: {"img_accs": [], "txt_accs": [], "count": 0})
    for r in rows_data:
        cat = r["category"]
        categories[cat]["count"] += 1
        if r["img_acc"] is not None:
            categories[cat]["img_accs"].append(r["img_acc"])
        if r["txt_acc"] is not None:
            categories[cat]["txt_accs"].append(r["txt_acc"])

    for cat in sorted(categories):
        data = categories[cat]
        n = data["count"]
        mean_img = sum(data["img_accs"]) / len(data["img_accs"]) if data["img_accs"] else None
        mean_txt = sum(data["txt_accs"]) / len(data["txt_accs"]) if data["txt_accs"] else None

        img_str = f"{mean_img:.1%}" if mean_img is not None else "—"
        txt_str = f"{mean_txt:.1%}" if mean_txt is not None else "—"
        if mean_img is not None and mean_txt is not None:
            gap_str = f"{mean_txt - mean_img:+.1%}"
        else:
            gap_str = "—"

        print(f"  {cat:<22} {n:>8} {img_str:>9} {txt_str:>9} {gap_str:>9}")

    print("└" + "─" * 67 + "┘")
    print()
