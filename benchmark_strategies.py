#!/usr/bin/env python3
"""Benchmark runner: evaluate all strategies on worst blind spot tasks.

Usage:
    # Generate data + run all strategies (requires local model running)
    python benchmark_strategies.py --api-base http://127.0.0.1:1234/v1 --model qwen3-vl-8b

    # Generate data only (no model needed)
    python benchmark_strategies.py --generate-only

    # Run strategies on existing data
    python benchmark_strategies.py --api-base http://127.0.0.1:1234/v1 --model qwen3-vl-8b --skip-generate

    # Run only specific strategies
    python benchmark_strategies.py --api-base ... --strategies best_of_n verify
"""

import argparse
import json
import sys
from pathlib import Path

from config import DATA_DIR, RESULTS_DIR, MODEL, API_BASE


# Tasks with the worst perceptual blind spots on Qwen3-VL-8B
BLIND_SPOT_TASKS = [
    "pie_chart",
    "colored_paths",
    "nested_squares",
    "hierarchy_depth",
    "realistic_table",
    "progress_bar",
    "scatter_plot",
    "text_degradation",
    "counting_grid",
]

ALL_STRATEGIES = [
    "baseline",
    "best_of_n",
    "crop_zoom",
    "verify",
    "decompose",
    "code_vision",
    "best_of_n_verify",
]

DEFAULT_SAMPLES = 20  # Per task for benchmarking


def generate_benchmark_data(tasks: list[str], n_samples: int):
    """Generate test manifests for each blind spot task."""
    from tasks import TASK_REGISTRY

    benchmark_dir = DATA_DIR / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    manifests = {}
    for task_name in tasks:
        if task_name not in TASK_REGISTRY:
            print(f"  WARNING: Task {task_name} not in registry, skipping")
            continue

        # Generate via CLI machinery
        class FakeArgs:
            pass
        args = FakeArgs()
        args.sweep = True
        args.n = n_samples
        args.n_per_config = None
        args.min_samples = n_samples
        args.max_per_config = n_samples
        args.max_total = n_samples
        args.prompt_variant = 1

        from cli import _generate_task
        config = TASK_REGISTRY[task_name]
        manifest_path = _generate_task(config, args)
        manifests[task_name] = manifest_path

    # Create combined manifest
    combined_path = benchmark_dir / "manifest.jsonl"
    total = 0
    with open(combined_path, "w") as out:
        for task_name, mp in manifests.items():
            with open(mp) as f:
                for line in f:
                    out.write(line)
                    total += 1

    print(f"\nCombined benchmark manifest: {total} samples → {combined_path}")
    return combined_path


def run_strategy(manifest_path: Path, strategy: str, model: str,
                 api_base: str, best_of_n: int = 5, workers: int = 3):
    """Run a single strategy and return the results path."""
    from harness import evaluate_manifest

    strategy_dir = RESULTS_DIR / "benchmark" / strategy
    results_path = strategy_dir / "results.jsonl"

    # Clear previous results for fresh comparison
    if results_path.exists():
        results_path.unlink()
    if results_path.with_suffix(".trace.jsonl").exists():
        results_path.with_suffix(".trace.jsonl").unlink()

    strategy_kwargs = {}
    if strategy in ("best_of_n", "best_of_n_verify"):
        strategy_kwargs["n"] = best_of_n

    actual_strategy = strategy if strategy != "baseline" else None

    print(f"\n{'='*60}")
    print(f"  Strategy: {strategy}")
    print(f"{'='*60}")

    evaluate_manifest(
        manifest_path=manifest_path,
        results_path=results_path,
        model=model,
        max_workers=workers,
        reasoning=False,
        api_base=api_base,
        strategy=actual_strategy,
        strategy_kwargs=strategy_kwargs,
    )

    return results_path


def print_comparison(results_paths: dict[str, Path]):
    """Print a comprehensive comparison across all strategies."""
    from analysis import load_results

    # Load all results
    all_results = {}
    for strategy, path in results_paths.items():
        if path.exists():
            all_results[strategy] = load_results(path)
        else:
            print(f"  WARNING: No results for {strategy} at {path}")

    if not all_results:
        print("No results to compare.")
        return

    # Get all tasks
    all_tasks = set()
    for results in all_results.values():
        for r in results:
            if not r["task_name"].endswith("_text"):
                all_tasks.add(r["task_name"])
    all_tasks = sorted(all_tasks)

    # Build accuracy table
    strategies = list(all_results.keys())

    # Header
    header = f"{'Task':<22}"
    for s in strategies:
        header += f" {s:>12}"
    header += f" {'best':>12} {'best_strategy':>15}"
    print(f"\n{header}")
    print("-" * len(header))

    task_bests = {}
    for task in all_tasks:
        row = f"{task:<22}"
        accs = {}
        for s in strategies:
            recs = [r for r in all_results[s] if r["task_name"] == task]
            if recs:
                acc = sum(1 for r in recs if r.get("correct")) / len(recs) * 100
                accs[s] = acc
                row += f" {acc:>11.0f}%"
            else:
                row += f" {'—':>12}"

        if accs:
            best_s = max(accs, key=accs.get)
            best_acc = accs[best_s]
            row += f" {best_acc:>11.0f}% {best_s:>15}"
            task_bests[task] = (best_s, best_acc)
        print(row)

    # Totals
    print("-" * len(header))
    row = f"{'MEAN':<22}"
    means = {}
    for s in strategies:
        recs = [r for r in all_results[s] if not r["task_name"].endswith("_text")]
        if recs:
            acc = sum(1 for r in recs if r.get("correct")) / len(recs) * 100
            means[s] = acc
            row += f" {acc:>11.1f}%"
        else:
            row += f" {'—':>12}"

    if means:
        best_s = max(means, key=means.get)
        row += f" {means[best_s]:>11.1f}% {best_s:>15}"
    print(row)

    # Summary
    if "baseline" in means:
        baseline_acc = means["baseline"]
        print(f"\nBaseline accuracy: {baseline_acc:.1f}%")
        for s in strategies:
            if s != "baseline" and s in means:
                delta = means[s] - baseline_acc
                marker = "↑" if delta > 0 else "↓" if delta < 0 else "→"
                print(f"  {s}: {means[s]:.1f}% ({delta:+.1f}p {marker})")

    # Per-task best strategy
    print(f"\nPer-task best strategies:")
    for task, (best_s, best_acc) in sorted(task_bests.items()):
        baseline_acc = 0
        recs = [r for r in all_results.get("baseline", []) if r["task_name"] == task]
        if recs:
            baseline_acc = sum(1 for r in recs if r.get("correct")) / len(recs) * 100
        delta = best_acc - baseline_acc
        print(f"  {task:<22} → {best_s} ({best_acc:.0f}%, {delta:+.0f}p vs baseline)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference-time strategies")
    parser.add_argument("--api-base", default=None,
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--model", default=None, help="Model name")
    parser.add_argument("--strategies", nargs="+", default=ALL_STRATEGIES,
                        choices=ALL_STRATEGIES,
                        help=f"Strategies to benchmark (default: all)")
    parser.add_argument("--tasks", nargs="+", default=BLIND_SPOT_TASKS,
                        help="Tasks to benchmark")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_SAMPLES,
                        help=f"Samples per task (default: {DEFAULT_SAMPLES})")
    parser.add_argument("--best-of-n", type=int, default=5,
                        help="N for best_of_n strategy (default: 5)")
    parser.add_argument("--workers", type=int, default=3,
                        help="Parallel workers (default: 3)")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate data, don't run strategies")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip data generation, use existing manifests")
    parser.add_argument("--compare-only", action="store_true",
                        help="Only compare existing results, don't run anything")
    args = parser.parse_args()

    if args.compare_only:
        results_paths = {}
        for s in args.strategies:
            p = RESULTS_DIR / "benchmark" / s / "results.jsonl"
            if p.exists():
                results_paths[s] = p
        print_comparison(results_paths)
        return

    # Generate data
    if not args.skip_generate:
        print("=" * 60)
        print("  Generating benchmark data")
        print("=" * 60)
        manifest_path = generate_benchmark_data(args.tasks, args.n_samples)
    else:
        manifest_path = DATA_DIR / "benchmark" / "manifest.jsonl"
        if not manifest_path.exists():
            print(f"Manifest not found: {manifest_path}")
            print("Run without --skip-generate first.")
            sys.exit(1)

    if args.generate_only:
        print("\nData generated. Run with --skip-generate to evaluate.")
        return

    # Need API base for running strategies
    api_base = args.api_base or API_BASE
    if not api_base:
        print("ERROR: --api-base required for evaluation (or set VLM_API_BASE)")
        sys.exit(1)

    model = args.model or MODEL

    # Run each strategy
    results_paths = {}
    for strategy in args.strategies:
        try:
            rp = run_strategy(
                manifest_path=manifest_path,
                strategy=strategy,
                model=model,
                api_base=api_base,
                best_of_n=args.best_of_n,
                workers=args.workers,
            )
            results_paths[strategy] = rp
        except Exception as e:
            print(f"  ERROR running {strategy}: {e}")

    # Print comparison
    print_comparison(results_paths)


if __name__ == "__main__":
    main()
