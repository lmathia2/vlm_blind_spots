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
