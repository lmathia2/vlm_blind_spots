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
- hierarchy_depth: 20% vs 60% (+40p) — systematic +1 overcount
- realistic_table: 20% vs 60% (+40p) — parser bug with LaTeX $ delimiters
- progress_bar: 60% vs 100% (+40p)
- scatter_plot: 60% vs 100% (+40p)
- text_degradation: 60% vs 100% (+40p)

## Phase 1: Core Strategies (DONE - commit 65e97f9)

- [x] strategies.py: baseline, best_of_n, crop_zoom, verify, best_of_n_verify
- [x] Wired into harness.py and cli.py
- [x] OpenAIVisionClient for local model support

## Phase 2: Advanced Strategies + Analysis (DONE - commit 5ea26ac)

- [x] decompose: task-specific sub-question plans with context accumulation
- [x] code_vision: sandboxed Python REPL for image analysis
- [x] analyze --compare for strategy comparison

## Phase 3: Benchmark Runner + Bug Fixes (DONE - commit 79b3292)

- [x] benchmark_strategies.py: automated runner for all strategies on blind spots
- [x] Fix exact_string parser for LaTeX $...$ delimiters
- [x] Generated 176 benchmark samples across 9 tasks

## Phase 4: Adaptive Strategy + Parser Hardening (DONE - commit 7cde02b)

- [x] adaptive strategy: routes each task to its best strategy
- [x] Hardened integer and mc4 parsers for LaTeX ${N}$ patterns
- [x] Refined hierarchy_depth prompts to fix +1 overcount ("rows not edges")
- [x] Task-specific verify prompts

### All strategies:
| Strategy | API Calls | Description |
|----------|-----------|-------------|
| baseline | 1 | Single-pass, current behavior |
| best_of_n | N | Majority voting at temp=0.7 |
| crop_zoom | 2-5 | Task-specific tile/crop, reask, aggregate |
| verify | 2 | Answer → task-specific re-examine → final |
| decompose | 2-3 | Multi-step sub-questions with context accumulation |
| code_vision | 2 | Model writes PIL/numpy code in sandboxed REPL |
| best_of_n_verify | N+1 | Majority vote then verification |
| adaptive | varies | Routes each task to its best strategy |

### Adaptive routing table:
| Task | Strategy | Rationale |
|------|----------|-----------|
| counting_grid | decompose | Count H/V lines separately |
| nested_squares | crop_zoom | Zoom center for inner squares |
| hierarchy_depth | verify | Catches +1 overcount bias |
| colored_paths | decompose | Identify paths then count |
| pie_chart | crop_zoom | Focus on slice regions |
| text_degradation | crop_zoom | Upscale degraded text |
| realistic_table | decompose | Extract structure first |
| scatter_plot | crop_zoom | Focus on cluster regions |
| progress_bar | crop_zoom | Focus on bar region |
| (unknown) | best_of_n | Safe default |

## Next Steps

- [ ] Run benchmark with Qwen3-VL-8B: `python benchmark_strategies.py --api-base http://... --model ...`
- [ ] Analyze results: `python benchmark_strategies.py --compare-only`
- [ ] Iterate on prompts/strategies based on actual performance data
