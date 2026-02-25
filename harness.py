"""Evaluation harness: VisionClient + parallel evaluation with resume support."""

import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import anthropic

from config import MODEL, TEMPERATURE, MAX_TOKENS, MAX_WORKERS, THINKING_BUDGET, API_BASE
from parsers import PARSER_REGISTRY
from scorers import SCORER_REGISTRY


class VisionClient:
    """Wraps the Anthropic API for vision tasks."""

    def __init__(self, model: str = MODEL, temperature: float = TEMPERATURE,
                 max_tokens: int = MAX_TOKENS, reasoning: bool = False):
        self.client = anthropic.Anthropic()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning = reasoning

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

        kwargs = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64_data}},
                    {"type": "text", "text": prompt},
                ],
            }],
        }

        if self.reasoning:
            kwargs["max_tokens"] = self.max_tokens + THINKING_BUDGET
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": THINKING_BUDGET}
            # temperature must not be set (or =1) when thinking is enabled
        else:
            kwargs["max_tokens"] = self.max_tokens
            kwargs["temperature"] = self.temperature

        t0 = time.time()
        response = self.client.messages.create(**kwargs)
        latency = time.time() - t0

        # Extract text and thinking from content blocks
        text_response = ""
        thinking_text = ""
        for block in response.content:
            if block.type == "text":
                text_response = block.text
            elif block.type == "thinking":
                thinking_text = block.thinking

        result = {
            "raw_response": text_response,
            "latency_s": round(latency, 2),
            "model": self.model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "reasoning_mode": self.reasoning,
        }
        if self.reasoning:
            result["thinking_text"] = thinking_text
        return result


class OpenAIVisionClient:
    """Wraps an OpenAI-compatible API (e.g., LM Studio) for vision tasks."""

    def __init__(self, model: str, base_url: str,
                 temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="lm-studio")
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
        data_uri = f"data:{media_type};base64,{b64_data}"

        t0 = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        latency = time.time() - t0

        text_response = response.choices[0].message.content or ""
        usage = response.usage

        return {
            "raw_response": text_response,
            "latency_s": round(latency, 2),
            "model": self.model,
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "reasoning_mode": False,
        }


def make_client(model: str, reasoning: bool = False,
                api_base: Optional[str] = None) -> VisionClient | OpenAIVisionClient:
    """Factory: return the right client based on api_base."""
    if api_base:
        return OpenAIVisionClient(model=model, base_url=api_base)
    return VisionClient(model=model, reasoning=reasoning)


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


def _evaluate_sample(client, sample: dict,
                     strategy_fn=None, strategy_kwargs: dict | None = None) -> dict:
    """Evaluate a single sample: query model, parse, score.

    If strategy_fn is provided, it handles the query+parse step.
    Otherwise falls back to the default single-pass behavior.
    """
    strategy_kwargs = strategy_kwargs or {}

    try:
        if strategy_fn is not None:
            # Strategy handles query + parse internally
            result = strategy_fn(client, sample, **strategy_kwargs)
        else:
            # Default single-pass behavior
            result = dict(sample)
            response = client.query(sample["image_path"], sample["prompt"])
            result.update(response)

            parser_fn = PARSER_REGISTRY.get(sample["parser"])
            if parser_fn:
                parsed = parser_fn(response["raw_response"])
                result["parsed_answer"] = parsed
            else:
                result["parsed_answer"] = None

        # Score the result (strategies set parsed_answer, scoring is uniform)
        scorer_fn = SCORER_REGISTRY.get(sample["scorer"])
        if scorer_fn and result.get("parsed_answer") is not None:
            scores = scorer_fn(result["parsed_answer"], sample["ground_truth"])
            result.update(scores)
        elif scorer_fn:
            scores = scorer_fn(None, sample["ground_truth"])
            result.update(scores)
        else:
            result["correct"] = False
            result["score"] = 0.0

    except Exception as e:
        result = dict(sample)
        result["error_message"] = str(e)
        result["correct"] = False
        result["score"] = 0.0
        result["parsed_answer"] = None

    return result


def _format_trace(result: dict, index: int, total: int) -> str:
    """Format a human-readable trace for one evaluated sample."""
    sid = result.get("sample_id", "?")
    task = result.get("task_name", "?")
    correct = result.get("correct", False)
    verdict = "\033[32mCORRECT\033[0m" if correct else "\033[31mWRONG\033[0m"

    prompt = result.get("prompt", "")
    # Collapse whitespace and truncate for display
    prompt_short = " ".join(prompt.split())
    if len(prompt_short) > 120:
        prompt_short = prompt_short[:117] + "..."

    raw = result.get("raw_response", "")
    raw_short = " ".join(raw.split())
    if len(raw_short) > 200:
        raw_short = raw_short[:197] + "..."

    parsed = result.get("parsed_answer")
    gt = result.get("ground_truth", "?")
    scorer = result.get("scorer", "?")

    # Build score detail string from scorer-specific fields
    details = []
    if "precision" in result and scorer == "set_match":
        details.append(f"P={result['precision']:.0%}")
        details.append(f"R={result['recall']:.0%}")
    detail_str = f"  ({', '.join(details)})" if details else ""

    tokens_in = result.get("input_tokens", "?")
    tokens_out = result.get("output_tokens", "?")
    latency = result.get("latency_s", "?")

    lines = [
        f"── [{index}/{total}] {sid} | {task} | {verdict}{detail_str}",
        f"   Prompt:    {prompt_short}",
    ]
    # Show thinking summary if present
    thinking = result.get("thinking_text", "")
    if thinking:
        think_short = " ".join(thinking.split())
        if len(think_short) > 150:
            think_short = think_short[:147] + "..."
        lines.append(f"   Thinking:  {think_short}")
    lines.extend([
        f"   Response:  {raw_short}",
        f"   Parsed: {parsed}  │  GT: {gt}  │  {tokens_in}→{tokens_out} tok  │  {latency}s",
    ])
    if result.get("error_message"):
        lines.append(f"   Error:     {result['error_message']}")

    return "\n".join(lines)


def evaluate_manifest(
    manifest_path: str | Path,
    results_path: str | Path,
    model: Optional[str] = None,
    max_workers: Optional[int] = None,
    reasoning: bool = False,
    api_base: Optional[str] = None,
    strategy: Optional[str] = None,
    strategy_kwargs: Optional[dict] = None,
) -> Path:
    """Evaluate all samples in a manifest JSONL, with resume support.

    Args:
        strategy: Name of inference strategy from STRATEGY_REGISTRY
                  (e.g., 'best_of_n', 'crop_zoom', 'verify').
                  None or 'baseline' uses default single-pass.
        strategy_kwargs: Extra kwargs passed to the strategy function
                         (e.g., {'n': 5} for best_of_n).

    Returns the path to the results file.
    """
    manifest_path = Path(manifest_path)
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    max_workers = max_workers or MAX_WORKERS
    strategy_kwargs = strategy_kwargs or {}

    # Resolve strategy function
    strategy_fn = None
    if strategy and strategy != "baseline":
        from strategies import STRATEGY_REGISTRY
        if strategy not in STRATEGY_REGISTRY:
            available = list(STRATEGY_REGISTRY.keys())
            raise ValueError(f"Unknown strategy '{strategy}'. Available: {available}")
        strategy_fn = STRATEGY_REGISTRY[strategy]

    client = make_client(model=model or MODEL, reasoning=reasoning, api_base=api_base)

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

    mode_str = " [REASONING]" if reasoning else ""
    strategy_str = f" [{strategy}]" if strategy and strategy != "baseline" else ""
    print(f"Evaluating {len(remaining)}/{len(samples)} samples{mode_str}{strategy_str} "
          f"(skipping {len(completed)} already done) with {max_workers} workers...")

    # Trace log sits next to the results file
    trace_path = results_path.with_suffix(".trace.jsonl")

    # When using multi-pass strategies, limit parallelism to avoid
    # overwhelming the backend (each sample makes multiple API calls)
    effective_workers = max_workers
    if strategy_fn is not None and max_workers > 3:
        effective_workers = min(max_workers, 3)
        print(f"  (limiting to {effective_workers} parallel workers for multi-pass strategy)")

    with open(results_path, "a") as out_f, open(trace_path, "a") as trace_f:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(
                    _evaluate_sample, client, s, strategy_fn, strategy_kwargs
                ): s
                for s in remaining
            }
            done_count = 0
            for future in as_completed(futures):
                result = future.result()
                done_count += 1

                # Write full result
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()

                # Write compact trace record (no image path clutter, no base64)
                trace_record = {
                    "sample_id": result.get("sample_id"),
                    "task_name": result.get("task_name"),
                    "prompt": result.get("prompt"),
                    "raw_response": result.get("raw_response"),
                    "parsed_answer": result.get("parsed_answer"),
                    "ground_truth": result.get("ground_truth"),
                    "correct": result.get("correct"),
                    "score": result.get("score"),
                    "parser": result.get("parser"),
                    "scorer": result.get("scorer"),
                    "latency_s": result.get("latency_s"),
                    "input_tokens": result.get("input_tokens"),
                    "output_tokens": result.get("output_tokens"),
                }
                # Include scorer-specific fields
                for key in ("precision", "recall", "error_message"):
                    if key in result:
                        trace_record[key] = result[key]
                # Include strategy fields
                for key in ("strategy", "strategy_n", "strategy_votes",
                            "strategy_final_changed", "strategy_verify_confirmed"):
                    if key in result:
                        trace_record[key] = result[key]
                # Include reasoning fields
                if "reasoning_mode" in result:
                    trace_record["reasoning_mode"] = result["reasoning_mode"]
                if "thinking_text" in result:
                    # Truncate thinking for trace
                    txt = result["thinking_text"]
                    trace_record["thinking_text"] = txt[:500] if len(txt) > 500 else txt
                trace_f.write(json.dumps(trace_record) + "\n")
                trace_f.flush()

                # Print trace to stdout
                print(_format_trace(result, done_count, len(remaining)))

    print(f"\nDone. Results: {results_path}")
    print(f"      Traces:  {trace_path}")
    return results_path
