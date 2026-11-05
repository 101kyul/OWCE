"""Unconditioned baseline: harder items are requested with a generic
"general_adversarial" tag instead of a diagnosed weakness tag. Reads the D0 results of a run file.

    python experiments/baseline_unconditioned.py [--bench ...] [--n 300] \\
        [--run results/results_n300.json] [--out results/baseline_unconditioned.json]
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset
from owce import EvalTask, Trajectory
from owce.tasks import TaskResult
from owce.generators import ClusterGenerator, Runner, failed_samples, output_text
from owce.generation import generate_cluster_batch
from owce.results import load_d0_from_run

os.environ.setdefault("INSPECT_LOG_DIR", str(ROOT / "logs" / "inspect"))

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
(ROOT / "logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(ROOT / "logs" / f"baseline_unconditioned_{RUN_ID}.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("baseline_unconditioned")

GENERATOR_MODEL = "openrouter/deepseek/deepseek-chat"
MAX_DEPTH = 2
SEED = 42
DEFAULT_N = 300

THRESHOLDS = {
    "gsm8k": 0.96, "math": 0.75, "gpqa": 0.49, "bbh": 0.72,
    "arc": 0.94, "truthfulqa": 0.80, "bbq": 0.92,
}

ALL_BENCHMARKS = ["gsm8k", "math", "gpqa", "bbh", "arc", "truthfulqa", "bbq"]

TARGET_MODELS = [
    ("openai/gpt-5.4",                                    "GPT-5.4"),
    ("anthropic/claude-sonnet-4-5-20250929",              "Claude-Sonnet-4.5"),
    ("anthropic/claude-haiku-4-5-20251001",               "Claude-Haiku-4.5"),
    ("openai/gpt-4o-mini",                                "GPT-4o-mini"),
    ("together/meta-llama/Llama-3.3-70B-Instruct-Turbo", "Llama-3.3-70B"),
    ("together/Qwen/Qwen2.5-7B-Instruct-Turbo",          "Qwen2.5-7B"),
]
MODEL_ID_MAP = {d: m for m, d in TARGET_MODELS}

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

BBQ_CATEGORIES = [
    "Age", "Disability_status", "Gender_identity", "Nationality",
    "Physical_appearance", "Race_ethnicity", "Race_x_SES", "Race_x_gender",
    "Religion", "SES", "Sexual_orientation",
]

RUN_PATH = ROOT / "results" / "results_n300.json"   # set by --run
RESULTS_PATH = ROOT / "results" / "baseline_unconditioned.json"   # set by --out


def _raw_task(name: str):
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
    raw = _raw_task(name)
    samples = list(raw.dataset)
    rng = random.Random(seed)
    rng.shuffle(samples)
    samples = samples[:n]
    return Task(
        dataset=MemoryDataset(samples=samples, name=f"{name}_n{n}_s{seed}", location="seeded"),
        solver=raw.solver, scorer=raw.scorer,
    )


def load_bbh_stratified(n_per_subtask: int, seed: int = SEED) -> Task:
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
            s = list(subtask.dataset)
            rng = random.Random(seed)
            rng.shuffle(s)
            all_samples.extend(s[:n_per_subtask])
            ok += 1
        except Exception as e:
            logger.warning(f"BBH subtask '{subset}' skip: {e}")
    full = bbh()
    return Task(
        dataset=MemoryDataset(all_samples, name=f"bbh_strat_{n_per_subtask}x{ok}", location="stratified"),
        solver=full.solver, scorer=full.scorer,
    )


def _bbq_counts_at_n(n: int) -> list[int]:
    base = n // 11
    rem  = n % 11
    return [base + (1 if i < rem else 0) for i in range(11)]


def load_bbq_stratified(n: int, seed: int = SEED) -> Task:
    from inspect_evals.bbq import bbq
    raw = bbq()
    by_cat = {}
    for s in list(raw.dataset):
        cat = (s.metadata or {}).get("category", "unknown")
        by_cat.setdefault(cat, []).append(s)
    counts = _bbq_counts_at_n(n)
    all_samples = []
    for cat, cnt in zip(BBQ_CATEGORIES, counts):
        samp = by_cat.get(cat, [])
        rng = random.Random(seed)
        rng.shuffle(samp)
        all_samples.extend(samp[:cnt])
    return Task(
        dataset=MemoryDataset(all_samples, name=f"bbq_strat_n{n}", location="stratified"),
        solver=raw.solver, scorer=raw.scorer,
    )


def load_benchmark(name: str, n: int, seed: int = SEED) -> tuple[Task, int]:
    if name == "bbh":
        n_per = math.ceil(n / 27)
        task = load_bbh_stratified(n_per_subtask=n_per, seed=seed)
        return task, n_per * len(BBH_SUBSETS)
    elif name == "bbq":
        return load_bbq_stratified(n=n, seed=seed), n
    else:
        return load_seeded(name, n=n, seed=seed), n


# Generic tag passed in place of a diagnosed weakness tag such as "negation_scope".
UNCONDITIONED_TAG = "general_adversarial"


@dataclass
class UnconditionedGenerator(ClusterGenerator):
    """ClusterGenerator that passes UNCONDITIONED_TAG to
    generate_cluster_batch instead of a diagnosed weakness tag, so the
    generator sees the same failure examples and prompt structure but no
    specific weakness information."""

    def generate_followup(
        self,
        parent_task: EvalTask,
        failure_result,
        max_candidates: int = 3,
    ) -> List[EvalTask]:
        print(f"      [unconditioned] {parent_task.id[:8]}")

        failed = failed_samples(failure_result)
        if not failed:
            print("      [unconditioned] No failed samples")
            return []

        print(f"      [unconditioned] {len(failed)} failures in one 'general_adversarial' cluster")

        cluster_items = []
        for s in failed:
            cluster_items.append({
                "question":    s.get("input", ""),
                "answer":      s.get("target", ""),
                "output":      output_text(s),
                "domain":      parent_task.meta.get("domain", "unknown"),
                "answer_type": parent_task.meta.get("answer_type", "text"),
                "weakness":    {"error_tags": [UNCONDITIONED_TAG]},
                "_id":         s.get("id", ""),
            })

        runner = Runner(self.model_name)

        generated_items = generate_cluster_batch(
            runner=runner,
            model_cfg=self.model_cfg,
            cluster_items=cluster_items,
            shared_weakness_tag=UNCONDITIONED_TAG,
            max_candidates=max_candidates,
            depth=parent_task.depth + 1,
            min_difficulty_increase=self.min_difficulty_increase,
        )

        print(f"      [unconditioned] Generated {len(generated_items)} verified items")

        followups = []
        for item in generated_items:
            task = self._item_to_task(item, parent_task)
            if task:
                task.weakness_tag = UNCONDITIONED_TAG
                followups.append(task)

        return followups


def save(data: dict):
    json.dump(data, open(RESULTS_PATH, "w"), indent=2, default=str)


def load_or_init() -> dict:
    if RESULTS_PATH.exists():
        logger.info(f"Checkpoint loaded: {RESULTS_PATH}")
        return json.load(open(RESULTS_PATH))
    return {"meta": {}, "d0_results": {}, "crosseval_12tag": []}


def run_expand_crosseval_unconditioned(
    bench_name: str, threshold: float,
    source_display: str, source_model_id: str,
    task: Task, limit: int,
    data: dict, n: int,
    d0_precomputed: dict,
):
    """Expand and cross-evaluate with the unconditioned generator."""
    key = "crosseval_12tag"
    logger.info(f"  [unconditioned] {source_display} x {bench_name.upper()} (N={n})")

    cfg = {"temperature": 0.5, "max_tokens": 32000}

    generator = UnconditionedGenerator(
        model_name=GENERATOR_MODEL, model_cfg=cfg, min_difficulty_increase=1.5,
    )

    seed_task = EvalTask(task=task, depth=0, weakness_tag=f"{bench_name}_unconditioned")
    trajectory = Trajectory(
        seed_tasks=[seed_task], generator=generator, threshold=threshold
    )

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
        limit_per_task=limit, max_candidates=5,
    )
    elapsed = time.time() - t0
    logger.info(f"  Nodes: {len(trajectory.nodes)}, Time: {elapsed:.0f}s")

    d1d2 = [(tid, t) for tid, t in trajectory.nodes.items() if t.depth > 0]
    if not d1d2:
        logger.info("  No D1/D2 tasks generated")
        return

    for tid, tnode in d1d2:
        node_scores = {}
        src = trajectory.results.get(tid)
        if src:
            node_scores[source_display] = {
                "score": src.score, "passed": src.score >= threshold, "is_source": True
            }

        for model_id, model_display in TARGET_MODELS:
            if model_display in node_scores:
                continue
            try:
                result = tnode.execute(model=model_id)
                node_scores[model_display] = {
                    "score": result.score, "passed": result.score >= threshold, "is_source": False
                }
                sym = "pass" if result.score >= 0.5 else "fail"
                logger.info(f"    {model_display:<22} {sym} {result.score:.3f}")
            except Exception as e:
                node_scores[model_display] = {"score": None, "passed": None, "is_source": False, "error": str(e)}
                logger.error(f"    {model_display:<22} ERR: {e}")

        data[key].append({
            "task_id": tid,
            "depth": tnode.depth,
            "weakness_tag": UNCONDITIONED_TAG,
            "benchmark": bench_name,
            "source_model": source_display,
            "generator": GENERATOR_MODEL,
            "conditioning": "unconditioned",
            "n": n,
            "seed": SEED,
            "scores": node_scores,
        })
        save(data)


def delta_summary(data: dict, n: int):
    print(f"\n{'-'*60}")
    print(f"Unconditioned baseline: transfer delta summary (N={n})")
    nodes = data.get("crosseval_12tag", [])
    for bench in ALL_BENCHMARKS:
        bench_nodes = [x for x in nodes if x.get("benchmark") == bench]
        if not bench_nodes:
            continue
        src, oth = [], []
        for node in bench_nodes:
            for s in node["scores"].values():
                if s.get("error") or s.get("score") is None:
                    continue
                (src if s.get("is_source") else oth).append(s["score"])
        if src and oth:
            delta = sum(oth)/len(oth) - sum(src)/len(src)
            print(f"  {bench:10s} delta={delta:+.3f}  nodes={len(bench_nodes)}")


def run_unconditioned_experiment(benchmarks: list[str], n: int = DEFAULT_N):
    d0_all = load_d0_from_run(RUN_PATH, benchmarks)
    if not d0_all:
        logger.error(f"no D0 results found in {RUN_PATH}")
        sys.exit(1)

    data = load_or_init()
    data["meta"] = {
        "conditioning": "unconditioned",
        "n": n, "seed": SEED,
        "generator": GENERATOR_MODEL,
        "run_id": RUN_ID,
        "benchmarks": benchmarks,
        "models": [d for _, d in TARGET_MODELS],
        "timestamp": datetime.now().isoformat(),
    }
    data["d0_results"] = {
        bench: {
            model: {
                "model_id": info["model_id"],
                "score": info["score"],
                "passed": info["passed"],
                "n_samples": info["n_samples"],
            }
            for model, info in d0_all[bench].items()
        }
        for bench in benchmarks if bench in d0_all
    }
    save(data)

    t_total = time.time()

    for bench_name in benchmarks:
        if bench_name not in d0_all:
            logger.warning(f"No D0 cache for {bench_name}, skipping")
            continue

        tau = THRESHOLDS[bench_name]
        d0_bench = d0_all[bench_name]
        task, effective_n = load_benchmark(bench_name, n=n, seed=SEED)

        failing = [(md, info) for md, info in d0_bench.items()
                   if not info.get("passed", False) and not info.get("error")]

        print(f"\n{'-'*60}")
        print(f"{bench_name.upper()}: tau={tau}, N={effective_n}, failing={[md for md, _ in failing]}")

        if not failing:
            print("  all models passed, skip")
            continue

        for model_display, info in failing:
            model_id = info.get("model_id") or MODEL_ID_MAP.get(model_display, "")
            if not model_id:
                continue

            existing = {x["source_model"] for x in data.get("crosseval_12tag", [])
                        if x.get("benchmark") == bench_name and x.get("n") == n}
            if model_display in existing:
                logger.info(f"  [unconditioned] {model_display} x {bench_name}: cached, skip")
                continue

            run_expand_crosseval_unconditioned(
                bench_name=bench_name, threshold=tau,
                source_display=model_display, source_model_id=model_id,
                task=task, limit=effective_n,
                data=data, n=n,
                d0_precomputed=info,
            )

    total_min = (time.time() - t_total) / 60
    logger.info(f"Done: {total_min:.0f} min")
    delta_summary(data, n)
    save(data)
    print(f"\nResults saved: {RESULTS_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bench", nargs="+", choices=ALL_BENCHMARKS, default=None,
                   help="Specific benchmarks (default: all)")
    p.add_argument("--n", type=int, default=DEFAULT_N,
                   help=f"Sample size (default: {DEFAULT_N})")
    p.add_argument("--run", type=Path, default=RUN_PATH, help="Run file whose D0 results seed the expansion")
    p.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = p.parse_args()
    RUN_PATH, RESULTS_PATH = args.run, args.out
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    benchmarks = args.bench or ALL_BENCHMARKS
    print("Unconditioned baseline")
    print(f"Benchmarks: {benchmarks}")
    print(f"N={args.n}, Generator={GENERATOR_MODEL}")
    print(f"Run file: {RUN_PATH}")
    print(f"Output:   {RESULTS_PATH}")

    run_unconditioned_experiment(benchmarks=benchmarks, n=args.n)
