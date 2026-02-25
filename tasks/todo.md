# Inference-Time Strategies for Closing VLM Perception Gap

## Goal
Implement inference-time strategies that improve Qwen3-VL-8B accuracy on perceptual blind spots
without retraining. Strategies compose with existing harness via a `--strategy` CLI flag.

## Approach
Create `strategies.py` with pluggable strategy functions that wrap `client.query()`.
Each strategy takes a client + sample dict and returns the same result format.
Wire into harness and CLI with `--strategy` and `--best-of-n` flags.

## Tasks

- [ ] 1. Create `strategies.py` with strategy framework + 3 strategies
  - `baseline`: current single-pass behavior
  - `best_of_n`: majority voting over N samples at temp>0
  - `crop_zoom`: tile/crop regions, reask, aggregate
  - `verify`: answer → verification pass → final answer
- [ ] 2. Modify `harness.py` to accept strategy parameter
- [ ] 3. Modify `cli.py` to add `--strategy` and `--best-of-n` flags
- [ ] 4. Write unit tests for all strategies
- [ ] 5. Verify full pipeline works end-to-end

## Files to modify
- `strategies.py` (new)
- `harness.py` (add strategy param to evaluate_manifest / _evaluate_sample)
- `cli.py` (add CLI flags)
- `tests/test_strategies.py` (new)

## Risks
- Crop-zoom needs task-specific crop logic — keep it generic with sensible defaults
- Best-of-N needs temp>0 which may not work with all backends
- Strategy overhead increases latency proportionally
