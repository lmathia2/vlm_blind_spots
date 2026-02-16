"""Evaluation harness: VisionClient + parallel evaluation with resume support."""

import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import anthropic

from config import MODEL, TEMPERATURE, MAX_TOKENS, MAX_WORKERS
from parsers import PARSER_REGISTRY
from scorers import SCORER_REGISTRY


class VisionClient:
    """Wraps the Anthropic API for vision tasks."""

    def __init__(self, model: str = MODEL, temperature: float = TEMPERATURE,
                 max_tokens: int = MAX_TOKENS):
        self.client = anthropic.Anthropic()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _encode_image(self, image_path: str) -> tuple[str, str]:
        """Read and base64-encode an image file. Returns (b64_data, media_type)."""
        path = Path(image_path)
        ext = path.suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_types.get(ext, "image/png")
        with open(path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        return data, media_type

    def query(self, image_path: str, prompt: str) -> dict:
        """Send an image + prompt to the model. Returns response dict."""
        b64_data, media_type = self._encode_image(image_path)
        t0 = time.time()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64_data}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        latency = time.time() - t0
        return {
            "raw_response": response.content[0].text,
            "latency_s": round(latency, 2),
            "model": self.model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }


def _load_completed_ids(results_path: Path) -> set[str]:
    """Load sample_ids already in results file for resume support."""
    completed = set()
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        completed.add(record["sample_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return completed


def _evaluate_sample(client: VisionClient, sample: dict) -> dict:
    """Evaluate a single sample: query model, parse, score."""
    result = dict(sample)
    try:
        response = client.query(sample["image_path"], sample["prompt"])
        result.update(response)

        parser_fn = PARSER_REGISTRY.get(sample["parser"])
        if parser_fn:
            parsed = parser_fn(response["raw_response"])
            result["parsed_answer"] = parsed
        else:
            result["parsed_answer"] = None

        scorer_fn = SCORER_REGISTRY.get(sample["scorer"])
        if scorer_fn and result["parsed_answer"] is not None:
            scores = scorer_fn(result["parsed_answer"], sample["ground_truth"])
            result.update(scores)
        elif scorer_fn and result["parsed_answer"] is None:
            scores = scorer_fn(None, sample["ground_truth"])
            result.update(scores)
        else:
            result["correct"] = False
            result["score"] = 0.0

    except Exception as e:
        result["error_message"] = str(e)
        result["correct"] = False
        result["score"] = 0.0
        result["parsed_answer"] = None

    return result


def evaluate_manifest(
    manifest_path: str | Path,
    results_path: str | Path,
    model: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> Path:
    """Evaluate all samples in a manifest JSONL, with resume support.

    Returns the path to the results file.
    """
    manifest_path = Path(manifest_path)
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    max_workers = max_workers or MAX_WORKERS

    client = VisionClient(model=model or MODEL)

    # Load manifest
    samples = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    # Resume: skip already-completed sample_ids
    completed = _load_completed_ids(results_path)
    remaining = [s for s in samples if s["sample_id"] not in completed]

    if not remaining:
        print(f"All {len(samples)} samples already evaluated. Nothing to do.")
        return results_path

    print(f"Evaluating {len(remaining)}/{len(samples)} samples "
          f"(skipping {len(completed)} already done) with {max_workers} workers...")

    with open(results_path, "a") as out_f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_evaluate_sample, client, s): s for s in remaining}
            done_count = 0
            for future in as_completed(futures):
                result = future.result()
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
                done_count += 1
                status = "CORRECT" if result.get("correct") else "WRONG"
                print(f"  [{done_count}/{len(remaining)}] {result.get('sample_id', '?')} "
                      f"→ {status} (parsed={result.get('parsed_answer')}, gt={result.get('ground_truth')})")

    print(f"Done. Results written to {results_path}")
    return results_path
