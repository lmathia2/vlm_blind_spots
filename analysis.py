"""Analysis: summary tables, accuracy curves, confusion matrices, failure examples."""

import json
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


def print_summary(results_path: str | Path):
    """Print accuracy summary table grouped by task_name."""
    results = load_results(results_path)
    if not results:
        print("No results found.")
        return

    df = pd.DataFrame(results)
    grouped = df.groupby("task_name")

    print(f"\n{'Task':<25} {'N':>5} {'Correct':>8} {'Accuracy':>9} {'Parse Fail':>11}")
    print("-" * 62)

    for task, group in sorted(grouped):
        n = len(group)
        correct = group["correct"].sum()
        accuracy = correct / n if n > 0 else 0
        parse_fails = group["parsed_answer"].isna().sum() if "parsed_answer" in group.columns else 0
        pf_rate = parse_fails / n if n > 0 else 0

        print(f"{task:<25} {n:>5} {int(correct):>8} {accuracy:>8.1%} {pf_rate:>10.1%}")

        # Show mean error for tasks with integer_distance scorer
        if "error" in group.columns and group["error"].notna().any():
            mean_err = group["error"].dropna().mean()
            print(f"  {'':>25} mean_error={mean_err:+.2f}")

    print(f"\n{'TOTAL':<25} {len(df):>5} {int(df['correct'].sum()):>8} "
          f"{df['correct'].mean():>8.1%}")


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
    output_dir = REPORT_ASSETS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Accuracy by task bar chart
    task_acc = df.groupby("task_name")["correct"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, max(4, len(task_acc) * 0.5)))
    task_acc.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Accuracy")
    ax.set_title("Accuracy by Task")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_by_task.png", dpi=150)
    plt.close(fig)
    print(f"Saved accuracy_by_task.png")

    # Per-task plots
    for task_name, group in df.groupby("task_name"):
        _plot_task_details(group, task_name, output_dir)


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
