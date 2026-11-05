# OWCE: Online Weakness-Conditioned Evaluation

OWCE evaluates a pool of models on seed benchmark items, tags each failure
with one of twelve weakness categories, groups the failures by tag, has a
held-out generator write harder follow-up items for each group, and
evaluates every generated item on every model. The run file records every
generated item with its tag and its score on every model.

## Setup

Python 3.11.

```bash
pip install -e .
export ANTHROPIC_API_KEY=...   # generator and Claude target models
export OPENAI_API_KEY=...      # GPT target models
export TOGETHER_API_KEY=...    # Llama, Qwen and DeepSeek models
export GOOGLE_API_KEY=...      # Gemini target model
export OPENROUTER_API_KEY=...  # experiments/baseline_unconditioned.py only
```

## Running

```bash
python run_experiment.py --bench gsm8k --n 10   # smoke test
python run_experiment.py                         # main run -> results/results_n300.json
```

| Option | Effect |
|--------|--------|
| `--bench B ...` | benchmarks to run (default: all seven) |
| `--n N` | seed items per benchmark (default 300) |
| `--generator M` | generator model string (default `anthropic/claude-opus-4-6`) |
| `--out NAME` | results file name under `results/` |
| `--sources weak-new --exclude-seeds-from RUN` | expand only the three weak sources, from seed items absent from `RUN` |
| `--sources strong` | expand the two strong models from every D0 failure |
| `--d0-from RUN` | reuse the D0 results of `RUN` |

Logs go to `logs/`, results to `results/`. A run resumes from its results
file: finished (benchmark, source) cells are skipped and D0 results reused.

## Experiments

Each script reads a run file with `--run` and writes to `--out`; its
docstring gives the command and its constants hold its own generator and
panel settings.

| Script | Purpose |
|--------|---------|
| `experiments/ablation_no_tag.py` | generation without a weakness tag |
| `experiments/ablation_permuted_tag.py` | generation with a random wrong tag |
| `experiments/baseline_unconditioned.py` | unconditioned generation (`openrouter/deepseek/deepseek-chat`, six targets, no Gemini) |
| `experiments/depth3_expansion.py` | expansion of the D2 items a source still fails |
| `experiments/verify_keys.py` | blind re-solving of every generated item by three solvers |
| `experiments/difficulty_rescore_generator.py` | difficulty re-scoring by the run's generator |
| `experiments/difficulty_rescore_panel.py` | difficulty re-scoring by GPT-5.4 and Gemini-2.5-Pro |
| `experiments/context_size_calibration.py` | targeting error as a function of the context size k |

## Configuration

| Parameter | Value |
|-----------|-------|
| Generator | `anthropic/claude-opus-4-6`, temperature 0.5, max_tokens 4096 |
| Seed items per benchmark | 300 (BBH 324, GPQA 198) |
| Depth | 2 |
| Seed | 42 |
| Failures per generator context | 4 |
| Difficulty gate | score >= 5 + depth + 1.5 |
| Follow-ups per failed parent | 20 |

| Target model | Model string |
|--------------|--------------|
| GPT-5.4 | `openai/gpt-5.4` |
| Claude Sonnet 4.5 | `anthropic/claude-sonnet-4-5-20250929` |
| Claude Haiku 4.5 | `anthropic/claude-haiku-4-5-20251001` |
| GPT-4o-mini | `openai/gpt-4o-mini` |
| Llama-3.3-70B | `together/meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| Qwen2.5-7B | `together/Qwen/Qwen2.5-7B-Instruct-Turbo` |
| Gemini-2.5-Pro | `google/gemini-2.5-pro` |

| Benchmark | τ |
|-----------|---|
| GSM8K | 0.96 |
| MATH | 0.75 |
| GPQA Diamond | 0.49 |
| BIG-Bench Hard | 0.72 |
| ARC Challenge | 0.94 |
| TruthfulQA | 0.80 |
| BBQ | 0.92 |

The twelve weakness tags and their patterns are in `owce/tagging.py`.

## Output

A run file holds `meta` (the run settings), `d0_results[benchmark][model]`
(score, pass flag, per-item samples) and `crosseval_12tag`, one entry per
generated item: `task_id`, `parent_id`, `depth`, `weakness_tag`,
`difficulty_score`, `benchmark`, `source_model`, `generator`, `n`, `seed`,
`probe_input`, `probe_target`, `probe_choices` and `source_item_ids` when
present, and `scores` for every model; a model whose evaluation failed is
recorded with `error` and no score. `completed` lists the finished
(benchmark, N, source) cells.

## Layout

```
run_experiment.py          pipeline
owce/tasks.py              Inspect task wrapper and result
owce/trajectory.py         weakness tree expansion
owce/tagging.py            weakness tagger
owce/clustering.py         grouping failures by tag
owce/generation.py         cluster-batch generation
owce/generation_single.py  per-item generation for isolated failures
owce/generators.py         ClusterGenerator, WeaknessClusterGenerator
owce/models.py             model calls through inspect-ai
owce/results.py            reading run files, Spearman correlation
experiments/               the scripts above
```
