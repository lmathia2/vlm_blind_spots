# Inference-Time Strategies for Closing VLM Perception Gap

## Goal
Implement inference-time strategies that improve Qwen3-VL-8B accuracy on perceptual blind spots
without retraining. Strategies compose with existing harness via a `--strategy` CLI flag.

## Phase 1: Core Strategies (DONE - commit 65e97f9)

- [x] Create `strategies.py` with 5 strategies: baseline, best_of_n, crop_zoom, verify, best_of_n_verify
- [x] Wire into `harness.py` (_evaluate_sample accepts strategy_fn + strategy_kwargs)
- [x] Add `--strategy` and `--best-of-n` CLI flags to `cli.py`
- [x] 42 unit tests in `tests/test_strategies.py`, 306 total passing
- [x] OpenAIVisionClient added for local model support (LM Studio / vLLM)

### What was built:
- **best_of_n**: Samples N responses at temp=0.7, majority votes on parsed answers
- **crop_zoom**: Task-specific crop configs (tile for grids, center-zoom for nested shapes/charts)
- **verify**: Two-pass answer→verify loop with confirmation detection
- **best_of_n_verify**: Composite — majority vote then verification pass
- **CLI**: `python cli.py evaluate --manifest ... --strategy best_of_n --best-of-n 5`

## Phase 2: Strategy Comparison Analysis + Refinements

- [ ] Add `analyze --compare` mode to compare strategy vs baseline results
- [ ] Refine crop_zoom prompts for worst tasks (pie_chart, hierarchy_depth)
- [ ] Add `structured_decomposition` strategy for multi-step reasoning
- [ ] Run full test suite after changes

## Phase 3: Code-Augmented Vision (RLM-style REPL)

- [ ] Implement sandboxed Python REPL for image analysis
- [ ] Add `code_vision` strategy that lets model write PIL/OpenCV code
- [ ] Target geometric tasks: counting_grid, nested_squares, edge_crossing
