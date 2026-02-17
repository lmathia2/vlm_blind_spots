"""Generate per-task failure example figures for the top 10 worst tasks.

For each task, produces report_haiku45/figures/fig_failure_example_{task}.png
containing 5 failure examples in a grid. Each example shows the original image,
prompt, ground truth, model answer, and error-type classification.

Usage:
    python report/generate_failure_cards.py [results_jsonl] [fig_dir]
"""
import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import defaultdict
from PIL import Image

RESULTS_FILE = sys.argv[1] if len(sys.argv) > 1 else "results_haiku45/results_filtered.jsonl"
FIG_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("report_haiku45/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

TOP_N_TASKS = 10
FAILURES_PER_TASK = 5


# ── Error-type classification heuristics ──

def classify_error(record):
    """Return a short error-type label based on record fields."""
    task = record["task_name"]
    gt = str(record.get("ground_truth", ""))
    ans = str(record.get("parsed_answer", "") or "")

    if ans == "" or ans == "None":
        return "Parse failure"

    # row_col tasks (check before numeric to avoid partial match on "25,8")
    if "," in gt and "," in ans:
        try:
            gt_parts = [int(x) for x in gt.split(",")]
            ans_parts = [int(x) for x in ans.split(",")]
            errors = [a - g for g, a in zip(gt_parts, ans_parts)]
            labels = []
            dims = ["rows", "cols"] if len(errors) == 2 else [f"d{i}" for i in range(len(errors))]
            for dim, err in zip(dims, errors):
                if err != 0:
                    labels.append(f"{dim} {'over' if err > 0 else 'under'} by {abs(err)}")
            if labels:
                return "; ".join(labels)
        except (ValueError, AttributeError):
            pass

    # Counting tasks: over vs undercount
    try:
        gt_num = float(gt.replace(",", "").strip())
        ans_num = float(ans.replace(",", "").strip())
        diff = ans_num - gt_num
        if diff > 0:
            return f"Overcount (+{diff:g})"
        elif diff < 0:
            return f"Undercount ({diff:g})"
    except (ValueError, AttributeError):
        pass

    # Text tasks: hallucination vs minor error
    if task in ("text_degradation", "dense_text", "rotated_text", "highlighted_text"):
        if gt and ans:
            common = sum(1 for a, b in zip(gt.lower(), ans.lower()) if a == b)
            ratio = common / max(len(gt), len(ans), 1)
            if ratio < 0.3:
                return "Hallucination"
            elif ratio < 0.8:
                return "Major char errors"
            else:
                return "Minor char error"

    # Set-based tasks
    if record.get("scorer") == "set_match":
        gt_set = set(x.strip() for x in gt.lower().split(","))
        ans_set = set(x.strip() for x in ans.lower().split(","))
        extra = ans_set - gt_set
        missing = gt_set - ans_set
        parts = []
        if extra:
            parts.append(f"extra: {','.join(sorted(extra))}")
        if missing:
            parts.append(f"missing: {','.join(sorted(missing))}")
        if parts:
            return "; ".join(parts)

    # Fallback
    if gt.lower() != ans.lower():
        gt_short = gt[:25] + "..." if len(gt) > 25 else gt
        ans_short = ans[:25] + "..." if len(ans) > 25 else ans
        return f"Wrong (GT={gt_short}, got={ans_short})"
    return "Unknown"


def is_artifact_failure(record):
    """Return True if this 'failure' is a measurement artifact, not a real model error."""
    task = record["task_name"]
    params = record.get("params", {})

    # pie_chart: another slice is within 5pp of the target, making visual
    # discrimination genuinely ambiguous (defensive check for legacy data)
    if task == "pie_chart":
        pcts = params.get("percentages", [])
        target_pct = params.get("target_pct", 0)
        others = [p for p in pcts if p != target_pct]
        if others and min(abs(p - target_pct) for p in others) <= 5:
            return True

    return False


def select_diverse_failures(records, n=5):
    """Pick up to n failures, preferring diversity in error types."""
    wrong = [r for r in records if not r.get("correct")]
    if not wrong:
        return []
    # Only keep records whose images exist
    wrong = [r for r in wrong if r.get("image_path") and os.path.exists(r["image_path"])]
    # Filter out measurement artifacts
    wrong = [r for r in wrong if not is_artifact_failure(r)]
    for r in wrong:
        r["_error_type"] = classify_error(r)
    # Round-robin across error types for diversity
    by_type = defaultdict(list)
    for r in wrong:
        by_type[r["_error_type"]].append(r)
    selected = []
    seen_ids = set()
    types_sorted = sorted(by_type.keys(), key=lambda t: -len(by_type[t]))
    while len(selected) < n:
        added_any = False
        for etype in types_sorted:
            if len(selected) >= n:
                break
            for c in by_type[etype]:
                if c["sample_id"] not in seen_ids:
                    selected.append(c)
                    seen_ids.add(c["sample_id"])
                    by_type[etype].remove(c)
                    added_any = True
                    break
        if not added_any:
            break
    return selected[:n]


def render_task_failures(task_name, acc, total, fail_count, failures, fig_dir):
    """Render a single composite figure with 5 failure examples for one task."""
    n = len(failures)
    if n == 0:
        return False

    display_name = task_name.replace("_", " ").title()

    # Layout: 5 rows, each row = [image | text panel]
    fig = plt.figure(figsize=(14, 4.2 * n))
    outer_gs = gridspec.GridSpec(n, 1, hspace=0.35)

    for i, record in enumerate(failures):
        inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[i],
                                                    width_ratios=[1, 1.3], wspace=0.08)

        gt = str(record.get("ground_truth", "N/A"))
        ans = str(record.get("parsed_answer", "N/A") or "N/A")
        prompt = record.get("prompt", "N/A")
        error_type = record.get("_error_type", classify_error(record))

        # Left: image
        ax_img = fig.add_subplot(inner_gs[0])
        img = Image.open(record["image_path"])
        ax_img.imshow(img)
        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_linewidth(1.5)
            spine.set_edgecolor("#DC2626")

        # Right: metadata text
        ax_txt = fig.add_subplot(inner_gs[1])
        ax_txt.axis("off")

        prompt_wrapped = textwrap.fill(prompt, width=60)
        if len(prompt_wrapped) > 250:
            prompt_wrapped = prompt_wrapped[:247] + "..."
        gt_display = gt if len(gt) <= 70 else gt[:67] + "..."
        ans_display = ans if len(ans) <= 70 else ans[:67] + "..."

        text_block = (
            f"Prompt:\n{prompt_wrapped}\n\n"
            f"Ground Truth:  {gt_display}\n"
            f"Model Answer:  {ans_display}\n\n"
            f"Error Type:  {error_type}"
        )

        ax_txt.text(
            0.02, 0.95, text_block,
            transform=ax_txt.transAxes,
            fontsize=8.5, fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FEF3C7",
                      edgecolor="#D97706", linewidth=1.2, alpha=0.95),
        )

    fig.suptitle(
        f"{display_name} — {acc:.0f}% accuracy ({fail_count} failures / {total} samples)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    out_path = fig_dir / f"fig_failure_example_{task_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out_path.name}")
    return True


def main():
    # Load results (image tasks only)
    tasks = {}
    with open(RESULTS_FILE) as f:
        for line in f:
            r = json.loads(line)
            t = r["task_name"]
            if t.endswith("_text"):
                continue
            tasks.setdefault(t, {"total": 0, "correct": 0, "failures": 0, "records": []})
            tasks[t]["total"] += 1
            if r.get("correct"):
                tasks[t]["correct"] += 1
            else:
                tasks[t]["failures"] += 1
            tasks[t]["records"].append(r)

    # Rank by accuracy ascending, take top 10
    ranked = sorted(
        [(name, d["correct"] / d["total"] * 100, d["total"], d["failures"], d["records"])
         for name, d in tasks.items() if d["failures"] > 0],
        key=lambda x: x[1]
    )[:TOP_N_TASKS]

    print(f"Generating failure figures for {len(ranked)} tasks...")

    for task_name, acc, total, fail_count, records in ranked:
        # Count real failures (excluding artifacts)
        real_failures = [r for r in records if not r.get("correct") and not is_artifact_failure(r)]
        real_fail_count = len(real_failures)
        real_acc = (total - real_fail_count) / total * 100
        if real_fail_count != fail_count:
            print(f"  {task_name}: {fail_count - real_fail_count} artifact(s) excluded "
                  f"({fail_count} raw -> {real_fail_count} real, acc {acc:.0f}% -> {real_acc:.0f}%)")
        failures = select_diverse_failures(records, FAILURES_PER_TASK)
        print(f"  {task_name}: {len(failures)} examples selected")
        render_task_failures(task_name, real_acc, total, real_fail_count, failures, FIG_DIR)

    print(f"\nDone. Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
