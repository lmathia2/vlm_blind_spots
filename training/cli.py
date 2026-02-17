"""CLI for training data generation and reward verification.

Usage:
    python -m training.cli generate --strategy all --output training_data/
    python -m training.cli verify --strategy direct --n 3
    python -m training.cli verify-reward --jsonl training_data/direct/samples.jsonl --n 5
"""

import argparse
import json
import sys
from pathlib import Path


def cmd_generate(args):
    """Generate SFT training data."""
    from training.sft_generate import generate_sft_dataset, generate_all

    output_dir = Path(args.output)

    if args.strategy == "all":
        generate_all(output_dir)
    else:
        generate_sft_dataset(
            output_dir,
            strategy=args.strategy,
            n_samples=args.n,
        )


def cmd_verify(args):
    """Print N samples for visual inspection."""
    from random import Random

    from training.sft_generate import generate_one_sample

    # Use seeds in 999K+ range (outside all training/eval ranges)
    base_seed = 999_000
    strategies = (
        ["direct", "intermediate_repr", "tool_use"]
        if args.strategy == "all"
        else [args.strategy]
    )

    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy}")
        print(f"{'='*60}")

        for i in range(args.n):
            seed = base_seed + i
            rng = Random(seed)
            sample = generate_one_sample(seed, strategy, rng)

            print(f"\n--- Sample {i+1} (seed={seed}) ---")
            print(f"Grid: {sample['metadata']['rows']}x{sample['metadata']['cols']}")
            print(f"Resolution: {sample['metadata']['resolution']}")
            print(f"Line width: {sample['metadata']['line_width']}")
            if sample.get("is_skip"):
                print("(tool-use skip — small grid)")
            print(f"\nPrompt: {sample['prompt']}")
            print(f"\nChain of thought:\n{sample['chain_of_thought']}")
            print(f"\nAnswer: {sample['answer']}")
            print(f"Ground truth: {sample['ground_truth']}")


def cmd_verify_reward(args):
    """Feed sample CoTs through reward functions, check they return 1.0."""
    from training.rewards import outcome_reward, process_reward, tool_use_reward

    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        if not jsonl_path.exists():
            print(f"File not found: {jsonl_path}")
            sys.exit(1)

        with open(jsonl_path) as f:
            samples = [json.loads(line) for line in f]

        if args.n:
            samples = samples[:args.n]
    else:
        # Generate fresh samples from verify seeds
        from random import Random
        from training.sft_generate import generate_one_sample

        n = args.n or 10
        samples = []
        for strategy in ["direct", "intermediate_repr", "tool_use"]:
            for i in range(n):
                seed = 999_000 + i
                rng = Random(seed)
                sample = generate_one_sample(seed, strategy, rng)
                samples.append({
                    "chain_of_thought": sample["chain_of_thought"],
                    "answer": sample["answer"],
                    "ground_truth": sample["ground_truth"],
                    "strategy": sample["strategy"],
                    "metadata": sample["metadata"],
                    "seed": sample["seed"],
                })

    reward_fns = {
        "outcome": outcome_reward,
        "process": process_reward,
        "tool_use": tool_use_reward,
    }

    n_checked = 0
    failures = []

    for sample in samples:
        # Build full response from CoT + answer
        response = sample["chain_of_thought"] + "\n\n" + sample["answer"]
        gt = sample["ground_truth"]
        meta = sample.get("metadata", {})

        results = {}
        for name, fn in reward_fns.items():
            score = fn(response, gt, meta)
            results[name] = score

        all_ok = all(v == 1.0 for v in results.values())
        n_checked += 1

        if not all_ok:
            failures.append({
                "seed": sample.get("seed"),
                "strategy": sample.get("strategy"),
                "ground_truth": gt,
                "rewards": results,
            })

    print(f"\nChecked {n_checked} samples")
    if failures:
        print(f"FAILURES: {len(failures)}")
        for f in failures:
            print(f"  seed={f['seed']} strategy={f['strategy']} "
                  f"gt={f['ground_truth']} rewards={f['rewards']}")
    else:
        print("All rewards returned 1.0")


def main():
    parser = argparse.ArgumentParser(
        description="Training data generation and reward verification"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate
    gen = subparsers.add_parser("generate", help="Generate SFT training data")
    gen.add_argument(
        "--strategy",
        choices=["direct", "intermediate_repr", "tool_use", "all"],
        default="all",
        help="Strategy to generate (default: all)",
    )
    gen.add_argument("--n", type=int, default=None, help="Number of samples")
    gen.add_argument(
        "--output", default="training_data",
        help="Output directory (default: training_data/)",
    )
    gen.set_defaults(func=cmd_generate)

    # verify
    ver = subparsers.add_parser("verify", help="Print samples for inspection")
    ver.add_argument(
        "--strategy",
        choices=["direct", "intermediate_repr", "tool_use", "all"],
        default="direct",
        help="Strategy to verify",
    )
    ver.add_argument("--n", type=int, default=3, help="Number of samples to show")
    ver.set_defaults(func=cmd_verify)

    # verify-reward
    vr = subparsers.add_parser(
        "verify-reward", help="Check reward functions on samples"
    )
    vr.add_argument("--jsonl", default=None, help="JSONL file to check")
    vr.add_argument(
        "--n", type=int, default=None,
        help="Number of samples (default: all from file, or 10 generated)",
    )
    vr.set_defaults(func=cmd_verify_reward)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
