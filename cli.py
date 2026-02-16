"""CLI: generate, evaluate, analyze, baseline subcommands."""

import argparse
import itertools
import json
import sys
import uuid
from pathlib import Path

from config import DATA_DIR, RESULTS_DIR, MODEL


def cmd_generate(args):
    """Generate images + manifest JSONL for a task (or all tasks)."""
    from tasks import TASK_REGISTRY

    if args.task == "all":
        tasks = sorted(TASK_REGISTRY.keys())
    elif args.task not in TASK_REGISTRY:
        print(f"Unknown task: {args.task}. Available: all, {list(TASK_REGISTRY.keys())}")
        sys.exit(1)
    else:
        tasks = [args.task]

    all_manifests = []
    for task_name in tasks:
        config = TASK_REGISTRY[task_name]
        manifest_path = _generate_task(config, args)
        all_manifests.append(manifest_path)

    # When generating all tasks, combine manifests automatically
    if args.task == "all":
        combined_dir = DATA_DIR / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)
        combined_path = combined_dir / "manifest.jsonl"
        total = 0
        with open(combined_path, "w") as out:
            for mp in all_manifests:
                with open(mp) as f:
                    for line in f:
                        out.write(line)
                        total += 1
        print(f"\nCombined manifest: {total} samples → {combined_path}")


def _generate_task(config: dict, args) -> Path:
    """Generate images + manifest for a single task. Returns manifest path."""
    render_fn = config["_render"]
    task_dir = DATA_DIR / config["task_name"]
    task_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = task_dir / "manifest.jsonl"

    prompt_variant = getattr(args, "prompt_variant", 1)
    prompt_key = "prompt_template" if prompt_variant == 1 else "prompt_template_v2"

    if args.sweep:
        param_combos = _sweep_combos(config)
        n_combos = len(param_combos)
        if args.n_per_config is not None:
            n_per = args.n_per_config
        else:
            # Auto-scale: at least min_samples_per_task total samples
            min_samples = args.min_samples
            max_per = args.max_per_config
            import math
            n_per = max(1, math.ceil(min_samples / n_combos))
            n_per = min(n_per, max_per)

        # Cap total samples per task; randomly sample combos if needed
        max_total = args.max_total
        total_planned = len(param_combos) * n_per
        if total_planned > max_total:
            import random as _rng
            _rng.seed(42)
            keep = max(1, max_total // n_per)
            _rng.shuffle(param_combos)
            param_combos = param_combos[:keep]
    else:
        param_combos = [config["default_params"]]
        n_per = args.n or 10

    import inspect
    sig = inspect.signature(render_fn)
    accepts_seed = "seed" in sig.parameters
    accepts_prompt_variant = "prompt_variant" in sig.parameters

    count = 0
    seen = set()
    with open(manifest_path, "w") as f:
        for params in param_combos:
            for i in range(n_per):
                sample_id = uuid.uuid4().hex[:8]
                kwargs = dict(params)
                if accepts_seed:
                    kwargs["seed"] = hash((str(params), i)) & 0x7FFFFFFF
                if accepts_prompt_variant:
                    kwargs["prompt_variant"] = prompt_variant

                img, ground_truth, metadata = render_fn(**kwargs)

                # Dedup only for deterministic tasks (no seed param)
                # Stochastic tasks produce unique outputs per call
                if not accepts_seed and n_per > 1:
                    dedup_key = (str(sorted(params.items())), ground_truth)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                img_filename = f"{sample_id}.png"
                img_path = task_dir / img_filename
                img.save(img_path)

                record = {
                    "sample_id": sample_id,
                    "task_name": config["task_name"],
                    "image_path": str(img_path),
                    "prompt": metadata.get("prompt", config[prompt_key]),
                    "ground_truth": ground_truth,
                    "parser": metadata.get("parser", config["parser"]),
                    "scorer": metadata.get("scorer", config["scorer"]),
                    "params": metadata,
                }
                if prompt_variant != 1:
                    record["prompt_variant"] = prompt_variant
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
    reasoning = getattr(args, "reasoning", False)
    evaluate_manifest(manifest_path, results_path, model=model,
                      max_workers=args.workers, reasoning=reasoning)


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
    gen.add_argument("--n-per-config", type=int, default=None,
                     help="Samples per sweep config (overrides --min-samples)")
    gen.add_argument("--min-samples", type=int, default=50,
                     help="Minimum samples per task in sweep mode (default: 50)")
    gen.add_argument("--max-per-config", type=int, default=100,
                     help="Maximum samples per sweep config (default: 100)")
    gen.add_argument("--max-total", type=int, default=100,
                     help="Maximum total samples per task; randomly samples configs if exceeded (default: 100)")
    gen.add_argument("--prompt-variant", type=int, default=1, choices=[1, 2],
                     help="Prompt variant (1=original, 2=rephrased)")
    gen.set_defaults(func=cmd_generate)

    # evaluate
    ev = subparsers.add_parser("evaluate", help="Evaluate a manifest")
    ev.add_argument("--manifest", required=True, help="Path to manifest JSONL")
    ev.add_argument("--model", default=None, help="Model override")
    ev.add_argument("--output", default=None, help="Output results path")
    ev.add_argument("--workers", type=int, default=None, help="Max parallel workers")
    ev.add_argument("--reasoning", action="store_true",
                     help="Enable extended thinking (reasoning mode)")
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
