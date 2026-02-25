# Inference-Time Strategies for Closing VLM Perception Gap

## Goal
Implement inference-time strategies that improve Qwen3-VL-8B accuracy on perceptual blind spots
without retraining. Strategies compose with existing harness via a `--strategy` CLI flag.

## Qwen3-VL-8B Baseline (N=5 per task, 70 tasks total)

Mean image accuracy: 70.3% | Mean text accuracy: 82.6% | Perception gap: +12.3p

Worst perceptual blind spots:
- pie_chart: 20% img vs 100% text (+80p gap)
- colored_paths: 40% vs 100% (+60p)
- nested_squares: 40% vs 100% (+60p)
- hierarchy_depth: 20% vs 60% (+40p)
- realistic_table: 20% vs 60% (+40p)
- progress_bar: 60% vs 100% (+40p)
- scatter_plot: 60% vs 100% (+40p)
- text_degradation: 60% vs 100% (+40p)

## Phase 1: Core Strategies (DONE - commit 65e97f9)

- [x] Create `strategies.py` with 5 strategies: baseline, best_of_n, crop_zoom, verify, best_of_n_verify
- [x] Wire into `harness.py` (_evaluate_sample accepts strategy_fn + strategy_kwargs)
- [x] Add `--strategy` and `--best-of-n` CLI flags to `cli.py`
- [x] 42 unit tests in `tests/test_strategies.py`, 306 total passing
- [x] OpenAIVisionClient added for local model support (LM Studio / vLLM)

## Phase 2: Advanced Strategies + Analysis (DONE - commit 5ea26ac)

- [x] Add `decompose` strategy with task-specific sub-question plans
- [x] Add `code_vision` strategy with sandboxed Python REPL
- [x] Add `analyze --compare` for baseline vs strategy comparison
- [x] 57 strategy tests, 321 total passing
- [x] Updated README with strategy documentation

### Strategies implemented:
| Strategy | API Calls | Description |
|----------|-----------|-------------|
| `baseline` | 1 | Single-pass, current behavior |
| `best_of_n` | N | Majority voting at temp=0.7 |
| `crop_zoom` | 2-5 | Task-specific tile/crop, reask, aggregate |
| `verify` | 2 | Answer → re-examine → final |
| `best_of_n_verify` | N+1 | Majority vote then verification |
| `decompose` | 2-3 | Multi-step sub-questions with context accumulation |
| `code_vision` | 2 | Model writes PIL/numpy code in sandboxed REPL |

## Next Steps

- [ ] Run larger baseline eval (50+ samples per task) for reliable measurements
- [ ] Run each strategy on the worst blind spot tasks and compare
- [ ] Iterate on prompts/crop configs based on actual results
- [ ] Consider combining strategies (e.g., decompose + best_of_n)
