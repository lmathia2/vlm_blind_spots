"""Generate visual inspection samples for fixed tasks.

Renders 5 sample images per task with prompt + ground truth overlaid,
for manual verification before running expensive API evaluations.

Usage:
    python report/generate_inspection_samples.py
"""
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

TASKS = ["realistic_table", "venn_diagram", "pie_chart"]
N_SAMPLES = 5
FIG_DIR = Path("report_haiku45/figures")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    from tasks import TASK_REGISTRY

    for task_name in TASKS:
        config = TASK_REGISTRY[task_name]
        render_fn = config["_render"]

        for i in range(N_SAMPLES):
            seed = 9000 + i  # deterministic, distinct from training seeds
            img, ground_truth, metadata = render_fn(seed=seed)
            prompt = metadata.get("prompt", "")

            fig = plt.figure(figsize=(10, 6))
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.05)

            # Left: rendered image
            ax_img = fig.add_subplot(gs[0])
            ax_img.imshow(img)
            ax_img.set_xticks([])
            ax_img.set_yticks([])
            ax_img.set_title("Rendered Image", fontsize=10)

            # Right: prompt + ground truth
            ax_txt = fig.add_subplot(gs[1])
            ax_txt.axis("off")
            prompt_wrapped = textwrap.fill(prompt, width=50)
            text_block = (
                f"Prompt:\n{prompt_wrapped}\n\n"
                f"Ground Truth:\n{ground_truth}"
            )
            ax_txt.text(
                0.05, 0.95, text_block,
                transform=ax_txt.transAxes,
                fontsize=9, fontfamily="monospace",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9",
                          edgecolor="#388E3C", linewidth=1.2, alpha=0.95),
            )

            display_name = task_name.replace("_", " ").title()
            fig.suptitle(f"{display_name} — Sample {i+1}", fontsize=12, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            out_path = FIG_DIR / f"inspect_{task_name}_{i}.png"
            fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            print(f"  -> {out_path}")

    print(f"\nDone. {len(TASKS) * N_SAMPLES} inspection images saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
