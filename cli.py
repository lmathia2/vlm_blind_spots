"""CLI: generate, evaluate, analyze, baseline subcommands."""

import argparse
import itertools
import json
import sys
import uuid
from pathlib import Path

from config import DATA_DIR, RESULTS_DIR, MODEL


def cmd_generate(args):
    """Generate images + manifest JSONL for a task."""
    from tasks import TASK_REGISTRY

    if args.task not in TASK_REGISTRY:
        print(f"Unknown task: {args.task}. Available: {list(TASK_REGISTRY.keys())}")
        sys.exit(1)

    config = TASK_REGISTRY[args.task]
    render_fn = config["_render"]
    task_dir = DATA_DIR / args.task
    task_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = task_dir / "manifest.jsonl"

    if args.sweep:
        param_combos = _sweep_combos(config)
        n_per = args.n_per_config or 1
    else:
        param_combos = [config["default_params"]]
        n_per = args.n or 10

    count = 0
    with open(manifest_path, "w") as f:
        for params in param_combos:
            for i in range(n_per):
                sample_id = uuid.uuid4().hex[:8]
                img, ground_truth, metadata = render_fn(**params)
                img_filename = f"{sample_id}.png"
                img_path = task_dir / img_filename
                img.save(img_path)

                record = {
                    "sample_id": sample_id,
                    "task_name": config["task_name"],
                    "image_path": str(img_path),
                    "prompt": metadata.get("prompt", config["prompt_template"]),
                    "ground_truth": ground_truth,
                    "parser": config["parser"],
                    "scorer": config["scorer"],
                    "params": metadata,
                }
                f.write(json.dumps(record) + "\n")
                count += 1

    print(f"Generated {count} samples → {manifest_path}")
    return manifest_path


def _sweep_combos(config: dict) -> list[dict]:
    """Generate all combinations of sweep axes, merged with default params."""
    sweep = config.get("sweep_axes", {})
    if not sweep:
        return [config["default_params"]]
    keys = list(sweep.keys())
    values = [sweep[k] for k in keys]
    combos = []
    for combo in itertools.product(*values):
        params = dict(config["default_params"])
        params.update(dict(zip(keys, combo)))
        combos.append(params)
    return combos


def cmd_evaluate(args):
    """Evaluate a manifest with the VLM."""
    from harness import evaluate_manifest

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    # Derive results path from manifest
    if args.output:
        results_path = Path(args.output)
    else:
        results_path = RESULTS_DIR / manifest_path.stem / "results.jsonl"

    model = args.model or MODEL
    evaluate_manifest(manifest_path, results_path, model=model,
                      max_workers=args.workers)


def cmd_analyze(args):
    """Print summary table and optionally generate plots."""
    from analysis import print_summary, generate_all_plots

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Results not found: {results_path}")
        sys.exit(1)

    print_summary(results_path)
    if args.plot:
        generate_all_plots(results_path)


def cmd_baseline(args):
    """Load BlindTest images, generate new tasks, evaluate everything."""
    from loaders.blindtest_loader import load_all_blindtest
    from harness import evaluate_manifest

    manifest_path = DATA_DIR / "baseline" / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    load_all_blindtest(manifest_path)

    results_path = RESULTS_DIR / "baseline" / "results.jsonl"
    model = args.model or MODEL
    evaluate_manifest(manifest_path, results_path, model=model,
                      max_workers=args.workers)

    from analysis import print_summary
    print_summary(results_path)


def main():
    parser = argparse.ArgumentParser(description="VLM Blind Spots Evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate
    gen = subparsers.add_parser("generate", help="Generate images + manifest")
    gen.add_argument("--task", required=True, help="Task name")
    gen.add_argument("--n", type=int, default=10, help="Number of samples (default mode)")
    gen.add_argument("--sweep", action="store_true", help="Sweep all parameter combinations")
    gen.add_argument("--n-per-config", type=int, default=1, help="Samples per sweep config")
    gen.set_defaults(func=cmd_generate)

    # evaluate
    ev = subparsers.add_parser("evaluate", help="Evaluate a manifest")
    ev.add_argument("--manifest", required=True, help="Path to manifest JSONL")
    ev.add_argument("--model", default=None, help="Model override")
    ev.add_argument("--output", default=None, help="Output results path")
    ev.add_argument("--workers", type=int, default=None, help="Max parallel workers")
    ev.set_defaults(func=cmd_evaluate)

    # analyze
    an = subparsers.add_parser("analyze", help="Analyze results")
    an.add_argument("--results", required=True, help="Path to results JSONL")
    an.add_argument("--plot", action="store_true", help="Generate plots")
    an.set_defaults(func=cmd_analyze)

    # baseline
    bl = subparsers.add_parser("baseline", help="Run baseline evaluation")
    bl.add_argument("--model", default=None, help="Model override")
    bl.add_argument("--workers", type=int, default=None, help="Max parallel workers")
    bl.set_defaults(func=cmd_baseline)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
