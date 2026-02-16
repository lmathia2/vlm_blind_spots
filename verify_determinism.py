"""Verify temp=0 determinism: run a small set of samples multiple times and compare responses."""

import json
import sys
from pathlib import Path

from config import DATA_DIR
from harness import VisionClient


def verify_determinism(n_samples: int = 10, n_repeats: int = 3):
    """Run n_samples across tasks n_repeats times each, report match rate."""
    # Collect a few samples from each available manifest
    manifests = sorted(DATA_DIR.glob("*/manifest.jsonl"))
    if not manifests:
        print("No manifests found. Generate tasks first.")
        sys.exit(1)

    all_samples = []
    for manifest_path in manifests:
        with open(manifest_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines[:2]:  # Take up to 2 per task
            all_samples.append(json.loads(line))
        if len(all_samples) >= n_samples:
            break

    all_samples = all_samples[:n_samples]
    print(f"Testing {len(all_samples)} samples x {n_repeats} repeats")

    client = VisionClient()
    results = []

    for i, sample in enumerate(all_samples):
        prompt = sample.get("prompt")
        image_path = sample.get("image_path")
        if not prompt or not Path(image_path).exists():
            print(f"  Skipping {sample['sample_id']}: missing prompt or image")
            continue

        responses = []
        for r in range(n_repeats):
            resp = client.query(image_path, prompt)
            responses.append(resp["raw_response"])

        all_same = len(set(responses)) == 1
        status = "MATCH" if all_same else "DIFFER"
        results.append({
            "sample_id": sample["sample_id"],
            "task_name": sample["task_name"],
            "match": all_same,
            "responses": responses,
        })

        print(f"  [{i+1}/{len(all_samples)}] {sample['task_name']}/{sample['sample_id']}: "
              f"{status}")
        if not all_same:
            for r_idx, resp in enumerate(responses):
                print(f"    Run {r_idx+1}: {resp[:100]}")

    n_match = sum(1 for r in results if r["match"])
    n_total = len(results)
    print(f"\nDeterminism: {n_match}/{n_total} samples identical across {n_repeats} runs "
          f"({n_match/n_total:.0%})" if n_total > 0 else "\nNo samples tested.")


if __name__ == "__main__":
    verify_determinism()
