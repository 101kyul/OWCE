"""OWCE pipeline: D0 evaluation, weakness tagging, clustering, generation
and cross-evaluation. `python run_experiment.py --help` lists the options;
the README gives the main invocations."""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset
from owce import EvalTask, Trajectory
from owce.tasks import TaskResult
from owce.generators import WeaknessClusterGenerator

BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
INSPECT_LOG_DIR = LOG_DIR / "inspect"
LOG_DIR.mkdir(exist_ok=True)
INSPECT_LOG_DIR.mkdir(exist_ok=True)
os.environ["INSPECT_LOG_DIR"] = str(INSPECT_LOG_DIR)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"experiment_{RUN_ID}.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("experiment")

GENERATOR_MODEL = "anthropic/claude-opus-4-6"
GENERATOR_CFG = {"temperature": 0.5, "max_tokens": 4096}
RESULTS_NAME = None
WEAK_SOURCES = ("GPT-4o-mini", "Llama-3.3-70B", "Qwen2.5-7B")
STRONG_SOURCES = ("GPT-5.4", "Claude-Haiku-4.5")
MAX_DEPTH = 2
SEED = 42                  # fixed seed; also what makes the N subsets nested
DEFAULT_N = 300
FOLLOWUP_BUDGET = 20       # generated items per failed parent

THRESHOLDS = {
    "gsm8k":      0.96,
    "math":       0.75,
    "gpqa":       0.49,
    "bbh":        0.72,
    "arc":        0.94,
    "truthfulqa": 0.80,
    "bbq":        0.92,
}

ALL_BENCHMARKS = ["gsm8k", "math", "gpqa", "bbh", "arc", "truthfulqa", "bbq"]

TARGET_MODELS = [
    ("openai/gpt-5.4",                                    "GPT-5.4"),
    ("anthropic/claude-sonnet-4-5-20250929",              "Claude-Sonnet-4.5"),
    ("anthropic/claude-haiku-4-5-20251001",               "Claude-Haiku-4.5"),
    ("openai/gpt-4o-mini",                                "GPT-4o-mini"),
    ("together/meta-llama/Llama-3.3-70B-Instruct-Turbo", "Llama-3.3-70B"),
    ("together/Qwen/Qwen2.5-7B-Instruct-Turbo",          "Qwen2.5-7B"),
    ("google/gemini-2.5-pro",                             "Gemini-2.5-Pro"),
]
MODEL_ID_MAP = {display: mid for mid, display in TARGET_MODELS}

# Official BIG-Bench Hard subtask names (27 tasks)
BBH_SUBSETS = [
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "dyck_languages", "formal_fallacies",
    "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
    "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "multistep_arithmetic_two", "navigate",
    "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
    "ruin_names", "salient_translation_error_detection", "snarks",
    "sports_understanding", "temporal_sequences",
    "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects", "web_of_lies", "word_sorting",
]

# Official BBQ category names (11 categories)
BBQ_CATEGORIES = [
    "Age", "Disability_status", "Gender_identity", "Nationality",
    "Physical_appearance", "Race_ethnicity", "Race_x_SES", "Race_x_gender",
    "Religion", "SES", "Sexual_orientation",
]


def _raw_task(name: str):
    """Load the raw benchmark task in its original order; load_seeded() does
    the seeded shuffle."""
    if name == "gsm8k":
        from inspect_evals.gsm8k import gsm8k; return gsm8k()
    if name == "math":
        from inspect_evals.math import math; return math(shuffle=False)
    if name == "gpqa":
        from inspect_evals.gpqa import gpqa_diamond; return gpqa_diamond()
    if name == "arc":
        from inspect_evals.arc import arc_challenge; return arc_challenge()
    if name == "truthfulqa":
        from inspect_evals.truthfulqa import truthfulqa; return truthfulqa(shuffle=False)
    if name == "bbq":
        from inspect_evals.bbq import bbq; return bbq()
    raise ValueError(f"Unknown benchmark: {name}")


def load_seeded(name: str, n: int, seed: int = SEED) -> Task:
    """Shuffle once with `seed`, then take the first `n` items, so the sample
    at a smaller N is a prefix of the sample at a larger N."""
    raw = _raw_task(name)
    samples = list(raw.dataset)
    rng = random.Random(seed)
    rng.shuffle(samples)
    samples = samples[:n]
    dataset = MemoryDataset(samples=samples, name=f"{name}_n{n}_s{seed}", location="seeded")
    return Task(dataset=dataset, solver=raw.solver, scorer=raw.scorer)


def load_bbh_stratified(n_per_subtask: int, seed: int = SEED) -> Task:
    """BBH with stratified sampling: n_per_subtask items from each of the 27
    subtasks, each shuffled with the same seed so the subsets nest across N."""
    from inspect_evals.bbh import bbh

    try:
        from inspect_evals.bbh.bbh import BBHDatasetRegistry
        subsets = BBHDatasetRegistry.get_all_subset_names()
    except (ImportError, AttributeError):
        subsets = BBH_SUBSETS

    all_samples, ok = [], 0
    for subset in subsets:
        try:
            subtask = bbh(subset_name=subset)
            samples = list(subtask.dataset)
            rng = random.Random(seed)
            rng.shuffle(samples)
            all_samples.extend(samples[:n_per_subtask])
            ok += 1
        except Exception as e:
            logger.warning(f"BBH subtask '{subset}' skip: {e}")

    logger.info(f"BBH stratified: {len(all_samples)} items ({ok} subtasks x {n_per_subtask})")
    full = bbh()
    return Task(
        dataset=MemoryDataset(all_samples, name=f"bbh_strat_{n_per_subtask}x{ok}", location="stratified"),
        solver=full.solver,
        scorer=full.scorer,
    )


def _bbq_counts_at_n(n: int) -> list[int]:
    """Per-category item counts for exactly N BBQ items across 11 categories.
    The remainder goes to the first k categories, in the same order for every
    N, so the subsets nest."""
    n_cats = len(BBQ_CATEGORIES)
    base = n // n_cats
    rem  = n % n_cats
    return [base + (1 if i < rem else 0) for i in range(n_cats)]


def load_bbq_stratified(n: int, seed: int = SEED) -> Task:
    """BBQ with stratified sampling: exactly N items across 11 categories."""
    from inspect_evals.bbq import bbq

    raw = bbq()
    all_raw_samples = list(raw.dataset)

    by_cat: dict[str, list] = {}
    for s in all_raw_samples:
        cat = (s.metadata or {}).get("category", "unknown")
        by_cat.setdefault(cat, []).append(s)

    counts = _bbq_counts_at_n(n)
    all_samples = []
    for cat, cnt in zip(BBQ_CATEGORIES, counts):
        samples = by_cat.get(cat, [])
        if len(samples) < cnt:
            logger.warning(f"BBQ category '{cat}': {len(samples)} items, {cnt} requested")
        rng = random.Random(seed)
        rng.shuffle(samples)
        all_samples.extend(samples[:cnt])

    logger.info(f"BBQ stratified: {len(all_samples)} items (N={n}, 11 cats, counts={counts[:4]}...)")
    return Task(
        dataset=MemoryDataset(
            all_samples,
            name=f"bbq_strat_n{n}",
            location="stratified",
        ),
        solver=raw.solver,
        scorer=raw.scorer,
    )


def load_benchmark(name: str, n: int, seed: int = SEED) -> tuple[Task, int]:
    """Load a benchmark with seeding. Returns (task, number of items loaded).
    BBH and BBQ are stratified; the others are a seeded prefix."""
    if name == "bbh":
        task = load_bbh_stratified(n_per_subtask=math.ceil(n / 27), seed=seed)
    elif name == "bbq":
        task = load_bbq_stratified(n=n, seed=seed)
    else:
        task = load_seeded(name, n=n, seed=seed)
    return task, len(task.dataset)


def results_path(n: int) -> Path:
    out_dir = BASE_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    return out_dir / (RESULTS_NAME or f"results_n{n}.json")


def save(data: dict, n: int):
    """Write the run file atomically."""
    path = results_path(n)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, path)


def load_or_init(n: int) -> dict:
    p = results_path(n)
    if p.exists():
        logger.info(f"  Checkpoint loaded: {p}")
        return json.load(open(p))
    return {"meta": {}, "d0_results": {}, "completed": [], "crosseval_12tag": []}


def _make_generator():
    """The twelve-tag cluster generator and its follow-up budget."""
    return WeaknessClusterGenerator(
        model_name=GENERATOR_MODEL, model_cfg=GENERATOR_CFG,
        min_cluster_size=2, min_difficulty_increase=1.5,
    ), FOLLOWUP_BUDGET


def run_expand_crosseval(
    bench_name: str, threshold: float,
    source_display: str, source_model_id: str,
    task: Task, limit: int,
    data: dict, n: int,
    d0_precomputed: dict,
):
    key = "crosseval_12tag"
    logger.info(f"  {source_display} x {bench_name.upper()} (N={n})")

    generator, budget = _make_generator()
    seed_task = EvalTask(task=task, depth=0, weakness_tag=f"{bench_name}_12tag")
    trajectory = Trajectory(seed_tasks=[seed_task], generator=generator, threshold=threshold)

    # Seed the trajectory with the stored D0 result; a re-run could land on the
    # other side of tau. expand() still generates follow-ups.
    samples = d0_precomputed.get("samples") or []
    score   = d0_precomputed.get("score", 0.0)
    pre_result = TaskResult(
        task_id=seed_task.id,
        score=score,
        scores={},
        metadata={},
        parent_id=None,
        depth=0,
        samples=samples,
    )
    trajectory.results[seed_task.id] = pre_result
    logger.info(f"  D0 pre-seeded: score={score:.3f} ({len(samples)} samples)")

    t0 = time.time()
    trajectory.expand(
        target_model=source_model_id, max_depth=MAX_DEPTH,
        limit_per_task=limit, max_candidates=budget,
    )
    elapsed = time.time() - t0
    logger.info(f"  Nodes: {len(trajectory.nodes)}, Time: {elapsed:.0f}s")

    d1d2 = [(tid, t) for tid, t in trajectory.nodes.items() if t.depth > 0]
    if not d1d2:
        logger.info("  No D1/D2 tasks generated")
        return

    for tid, tnode in d1d2:
        node_scores = {}
        probe_input  = None
        probe_target = None
        probe_choices = tnode.meta.get("choices", None)
        source_item_ids = tnode.meta.get("source_item_ids", None)
        src = trajectory.results.get(tid)
        if src:
            node_scores[source_display] = {"score": src.score, "passed": src.score >= threshold, "is_source": True}
            if src.samples:
                probe_input  = src.samples[0].get("input")
                probe_target = src.samples[0].get("target")

        for model_id, model_display in TARGET_MODELS:
            if model_display in node_scores:
                continue
            try:
                result = tnode.execute(model=model_id, threshold=threshold)
                node_scores[model_display] = {
                    "score": result.score, "passed": result.score >= threshold, "is_source": False,
                }
                if probe_input is None and result.samples:
                    probe_input  = result.samples[0].get("input")
                    probe_target = result.samples[0].get("target")
                sym = "pass" if result.score >= 0.5 else "fail"
                logger.info(f"    {model_display:<22} {sym} {result.score:.3f}")
            except Exception as e:
                node_scores[model_display] = {"score": None, "passed": None, "is_source": False, "error": str(e)}
                logger.error(f"    {model_display:<22} ERR: {e}")

        entry = {
            "task_id": tid, "parent_id": tnode.parent_id, "depth": tnode.depth,
            "weakness_tag": tnode.weakness_tag or f"{bench_name}_12tag",
            "difficulty_score": tnode.meta.get("difficulty_score"),
            "benchmark": bench_name, "source_model": source_display,
            "generator": GENERATOR_MODEL, "n": n, "seed": SEED, "scores": node_scores,
            "probe_input": probe_input,
            "probe_target": probe_target,
        }
        if probe_choices:
            entry["probe_choices"] = probe_choices
        if source_item_ids:
            entry["source_item_ids"] = source_item_ids
        data[key].append(entry)
        save(data, n)


def delta_summary(data: dict, n: int):
    print(f"\n{'-'*60}")
    print(f"Transfer delta summary (N={n})")
    for bench in ALL_BENCHMARKS:
        nodes = [x for x in data.get("crosseval_12tag", []) if x.get("benchmark") == bench]
        if not nodes:
            continue
        src, oth = [], []
        for node in nodes:
            for s in node["scores"].values():
                if s.get("error") or s.get("score") is None:
                    continue
                (src if s.get("is_source") else oth).append(s["score"])
        if src and oth:
            delta = sum(oth)/len(oth) - sum(src)/len(src)
            print(f"  {bench:10s} delta={delta:+.3f}  nodes={len(nodes)}")


def run_fresh_d0(bench_name: str, task: Task, effective_n: int, tau: float,
                 models=None) -> dict:
    """Run the D0 evaluation at the given N for `models` (default: all
    TARGET_MODELS). Returns dict[model_display -> {model_id, score, passed,
    n_samples, samples}]; a model whose eval failed carries "error" instead."""
    d0 = {}
    for model_id, model_display in (models or TARGET_MODELS):
        print(f"  D0  {model_display:<22} {bench_name.upper()} (N={effective_n})", flush=True)
        try:
            seed_task = EvalTask(task=task, depth=0, weakness_tag=f"{bench_name}_d0")
            result = seed_task.execute(model=model_id, threshold=tau)
            passed = result.score >= tau
            sym = "pass" if passed else "fail"
            logger.info(f"  D0 {model_display}: {result.score:.3f} ({sym})")
            d0[model_display] = {
                "model_id": model_id,
                "score": result.score,
                "passed": passed,
                "n_samples": len(result.samples),
                "samples": result.samples,
            }
        except Exception as e:
            logger.error(f"  D0 {model_display} ERROR: {e}")
            d0[model_display] = {
                "model_id": model_id,
                "score": None,
                "passed": None,
                "n_samples": 0,
                "error": str(e),
            }
    return d0


def _load_seed_ids(path: Path) -> dict:
    """(benchmark, model) -> seed item ids of another run's D0, to exclude."""
    other = json.load(open(path))
    out = {}
    for b, per in other.get("d0_results", {}).items():
        for m, r in per.items():
            out[(b, m)] = {sm["id"] for sm in r.get("samples", [])}
    return out


def run_condition(n: int, benchmarks: list[str], sources: str = "all",
                  exclude_seeds_from: Path | None = None, d0_from: Path | None = None):
    """The pipeline for one seed size N.

    Args:
        sources: "all" expands every model below tau; "weak-new" expands only
                 the three weak sources and only from seed items absent from
                 the run named by exclude_seeds_from; "strong" expands the two
                 strong models from every D0 failure, ignoring tau.
        exclude_seeds_from: run file whose D0 seed items are excluded.
        d0_from: run file whose D0 results are reused instead of re-running D0.
    """
    excluded = _load_seed_ids(exclude_seeds_from) if exclude_seeds_from else {}
    d0_reuse = json.load(open(d0_from)).get("d0_results", {}) if d0_from else {}
    d0_models = [(mid, md) for mid, md in TARGET_MODELS
                 if sources != "weak-new" or md in WEAK_SOURCES]
    logger.info(f"N = {n}  (seed={SEED})")

    data = load_or_init(n)
    data.setdefault("completed", [])
    data["meta"] = {
        "n": n, "seed": SEED,
        "generator": GENERATOR_MODEL, "generator_cfg": GENERATOR_CFG,
        "run_id": RUN_ID,
        "benchmarks": benchmarks,
        "models": [d for _, d in TARGET_MODELS],
        "timestamp": datetime.now().isoformat(),
        "sources": sources,
    }

    t_total = time.time()

    for bench_name in benchmarks:
        tau = THRESHOLDS[bench_name]

        task, effective_n = load_benchmark(bench_name, n=n, seed=SEED)

        logger.info(f"{bench_name.upper()} D0 (tau={tau}, N={effective_n})")

        # D0 precedence: a checkpointed or reused D0 result per model, then a
        # fresh run for the models that have none or whose eval failed.
        cached_d0 = data.get("d0_results", {}).get(bench_name, {}) or d0_reuse.get(bench_name, {})
        d0_bench = {md: v for md, v in cached_d0.items()
                    if not v.get("error") and v.get("n_samples") == effective_n}
        missing = [(mid, md) for mid, md in d0_models if md not in d0_bench]
        if d0_bench:
            logger.info(f"  D0 {bench_name}: cached for {sorted(d0_bench)}")
        if missing:
            d0_bench.update(run_fresh_d0(bench_name, task, effective_n, tau, models=missing))
        data.setdefault("d0_results", {})[bench_name] = d0_bench
        save(data, n)

        if sources == "strong":
            failing = [(md, info) for md, info in d0_bench.items()
                       if md in STRONG_SOURCES and not info.get("error")]
        elif sources == "weak-new":
            failing = []
            for md, info in d0_bench.items():
                if md not in WEAK_SOURCES or info.get("passed", False) or info.get("error"):
                    continue
                old = excluded.get((bench_name, md), set())
                info2 = dict(info)
                info2["samples"] = [sm for sm in (info.get("samples") or []) if sm.get("id") not in old]
                logger.info(f"  {md}: {len(info2['samples'])} new-seed samples of {len(info.get('samples') or [])}")
                failing.append((md, info2))
        else:
            failing = [(md, info) for md, info in d0_bench.items()
                       if not info.get("passed", False) and not info.get("error")]
        if not failing:
            logger.info(f"  all models passed tau={tau} at N={n}, skip expansion")
            continue

        logger.info(f"  {len(failing)} failing: {[md for md, _ in failing]}")

        for model_display, info in failing:
            model_id = info.get("model_id") or MODEL_ID_MAP.get(model_display, "")
            if not model_id:
                continue

            # A (benchmark, N, source) cell is skipped once its expansion and
            # cross-evaluation finished.
            cell = [bench_name, n, model_display]
            if cell in data["completed"]:
                logger.info(f"  {model_display} x {bench_name}: completed earlier, skip")
                continue
            data["crosseval_12tag"] = [x for x in data["crosseval_12tag"]
                                       if not (x.get("benchmark") == bench_name and x.get("n") == n
                                               and x.get("source_model") == model_display)]

            # "strong": every D0 miss counts as a failure; probe scores are 0/1,
            # so their pass flags are unaffected by threshold=1.0.
            run_expand_crosseval(
                bench_name=bench_name,
                threshold=1.0 if sources == "strong" else tau,
                source_display=model_display, source_model_id=model_id,
                task=task, limit=effective_n,
                data=data, n=n,
                d0_precomputed=info,
            )
            data["completed"].append(cell)
            save(data, n)

    total_min = (time.time() - t_total) / 60
    logger.info(f"N={n} done: {total_min:.0f} min")
    delta_summary(data, n)
    save(data, n)
    return data


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", nargs="+", choices=ALL_BENCHMARKS,
                   default=None, help="Specific benchmarks (default: all)")
    p.add_argument("--n", type=int, default=DEFAULT_N,
                   help=f"Seed items per benchmark (default: {DEFAULT_N})")
    p.add_argument("--generator", default=GENERATOR_MODEL,
                   help=f"Generator model (inspect-ai model string; default {GENERATOR_MODEL})")
    p.add_argument("--out", default=None,
                   help="Results file name under results/ (default results_n{N}.json)")
    p.add_argument("--sources", choices=["all", "weak-new", "strong"], default="all",
                   help="Which models to expand from: all below tau; the three weak sources on "
                        "new seeds only; or the two strong models from every D0 failure")
    p.add_argument("--exclude-seeds-from", type=Path, default=None,
                   help="Run file whose D0 seed items are excluded (with --sources weak-new)")
    p.add_argument("--d0-from", type=Path, default=None,
                   help="Run file whose D0 results are reused instead of re-running D0")
    args = p.parse_args()
    if args.sources == "weak-new" and not args.exclude_seeds_from:
        p.error("--sources weak-new requires --exclude-seeds-from")
    return args


def main():
    global GENERATOR_MODEL, RESULTS_NAME
    args = parse_args()
    benchmarks = args.bench or ALL_BENCHMARKS
    GENERATOR_MODEL = args.generator
    RESULTS_NAME = args.out

    logger.info(f"OWCE run_id={RUN_ID}")
    logger.info(f"Benchmarks: {benchmarks}")

    run_condition(args.n, benchmarks, sources=args.sources,
                  exclude_seeds_from=args.exclude_seeds_from, d0_from=args.d0_from)
    print(f"\nResults: {results_path(args.n)}")


if __name__ == "__main__":
    main()
