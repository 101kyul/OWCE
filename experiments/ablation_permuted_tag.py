"""Conditioning ablation, permuted tag: the weakness-tag line stays in the
prompt, but the diagnosed tag is replaced by a random wrong tag. Reads the D0 results
of a run file and cross-evaluates the generated items on the panel.

    python experiments/ablation_permuted_tag.py [--bench gpqa math bbh] [--n 300] \\
        [--run results/results_n300.json] [--out results/ablation_permuted_tag.json]
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict

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
from owce.tagging import analyze, WEAKNESS_KEYS
from owce.results import load_d0_from_run

os.environ.setdefault("INSPECT_LOG_DIR", str(ROOT / "logs" / "inspect"))

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
(ROOT / "logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            ROOT / "logs" / f"ablation_permuted_tag_{RUN_ID}.log", encoding="utf-8"
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("ablation_permuted_tag")

GENERATOR_MODEL = "anthropic/claude-opus-4-6"
MAX_DEPTH = 2
SEED = 42
DEFAULT_N = 300

THRESHOLDS = {
    "gsm8k": 0.96,
    "math": 0.75,
    "gpqa": 0.49,
    "bbh": 0.72,
    "arc": 0.94,
    "truthfulqa": 0.80,
    "bbq": 0.92,
}
ALL_BENCHMARKS = ["gsm8k", "math", "gpqa", "bbh", "arc", "truthfulqa", "bbq"]

TARGET_MODELS = [
    ("openai/gpt-5.4", "GPT-5.4"),
    ("anthropic/claude-sonnet-4-5-20250929", "Claude-Sonnet-4.5"),
    ("anthropic/claude-haiku-4-5-20251001", "Claude-Haiku-4.5"),
    ("openai/gpt-4o-mini", "GPT-4o-mini"),
    ("together/meta-llama/Llama-3.3-70B-Instruct-Turbo", "Llama-3.3-70B"),
    ("together/Qwen/Qwen2.5-7B-Instruct-Turbo", "Qwen2.5-7B"),
    ("google/gemini-2.5-pro", "Gemini-2.5-Pro"),
]
MODEL_ID_MAP = {d: m for m, d in TARGET_MODELS}

BBH_SUBSETS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]
BBQ_CATEGORIES = [
    "Age",
    "Disability_status",
    "Gender_identity",
    "Nationality",
    "Physical_appearance",
    "Race_ethnicity",
    "Race_x_SES",
    "Race_x_gender",
    "Religion",
    "SES",
    "Sexual_orientation",
]

RUN_PATH = ROOT / "results" / "results_n300.json"  # set by --run
RESULTS_PATH = ROOT / "results" / "ablation_permuted_tag.json"  # set by --out


def _raw_task(name):
    if name == "gsm8k":
        from inspect_evals.gsm8k import gsm8k

        return gsm8k()
    if name == "math":
        from inspect_evals.math import math

        return math(shuffle=False)
    if name == "gpqa":
        from inspect_evals.gpqa import gpqa_diamond

        return gpqa_diamond()
    if name == "arc":
        from inspect_evals.arc import arc_challenge

        return arc_challenge()
    if name == "truthfulqa":
        from inspect_evals.truthfulqa import truthfulqa

        return truthfulqa(shuffle=False)
    if name == "bbq":
        from inspect_evals.bbq import bbq

        return bbq()
    raise ValueError(f"Unknown: {name}")


def _bbq_counts_at_n(n):
    base, rem = n // 11, n % 11
    return [base + (1 if i < rem else 0) for i in range(11)]


def load_benchmark(name, n, seed=SEED):
    if name == "bbh":
        from inspect_evals.bbh import bbh

        n_per = math.ceil(n / 27)
        samples = []
        for subset in BBH_SUBSETS:
            try:
                st = bbh(subset_name=subset)
                s = list(st.dataset)
                random.Random(seed).shuffle(s)
                samples.extend(s[:n_per])
            except Exception as e:
                logger.warning(f"BBH subtask '{subset}' skip: {e}")
        full = bbh()
        return Task(
            dataset=MemoryDataset(
                samples, name=f"bbh_{n_per}x{len(BBH_SUBSETS)}", location="stratified"
            ),
            solver=full.solver,
            scorer=full.scorer,
        ), n_per * len(BBH_SUBSETS)
    if name == "bbq":
        from inspect_evals.bbq import bbq

        raw = bbq()
        by_cat = {}
        for s in raw.dataset:
            c = (s.metadata or {}).get("category", "unknown")
            by_cat.setdefault(c, []).append(s)
        counts = _bbq_counts_at_n(n)
        out = []
        for cat, cnt in zip(BBQ_CATEGORIES, counts):
            s = by_cat.get(cat, [])
            random.Random(seed).shuffle(s)
            out.extend(s[:cnt])
        return (
            Task(
                dataset=MemoryDataset(out, name=f"bbq_n{n}", location="stratified"),
                solver=raw.solver,
                scorer=raw.scorer,
            ),
            n,
        )
    raw = _raw_task(name)
    s = list(raw.dataset)
    random.Random(seed).shuffle(s)
    s = s[:n]
    return (
        Task(
            dataset=MemoryDataset(s, name=f"{name}_n{n}", location="seeded"),
            solver=raw.solver,
            scorer=raw.scorer,
        ),
        n,
    )


def _pick_permuted_tag(diagnosed_tag: str, rng: random.Random) -> str:
    """Pick a random taxonomy tag other than the diagnosed one."""
    candidates = [t for t in WEAKNESS_KEYS if t != diagnosed_tag]
    return rng.choice(candidates)


def _diagnose_dominant_tag(failed_samples: List[Dict]) -> str:
    """Run tagging.analyze() on every failed sample, as the pipeline does, and
    return the most frequent primary tag."""
    tag_counts: Dict[str, int] = {}
    for s in failed_samples:
        question = s.get("question", s.get("input", ""))
        result = analyze(question)
        primary = result.error_tags[0] if result.error_tags else "inference_chain"
        tag_counts[primary] = tag_counts.get(primary, 0) + 1
    if not tag_counts:
        return "inference_chain"
    return max(tag_counts, key=tag_counts.get)


@dataclass
class PermutedTagGenerator(ClusterGenerator):
    """One cluster of all failures per parent, prompted with the cluster
    prompt of the main pipeline but with the dominant diagnosed tag replaced
    by a random wrong taxonomy tag."""

    _rng: random.Random = field(default=None, init=False)

    def __post_init__(self):
        self._rng = random.Random(SEED)

    def generate_followup(self, parent_task, failure_result, max_candidates=3):
        failed = failed_samples(failure_result)
        if not failed:
            return []

        cluster_items = []
        for s in failed:
            cluster_items.append(
                {
                    "question": s.get("input", ""),
                    "answer": s.get("target", ""),
                    "output": output_text(s),
                    "domain": parent_task.meta.get("domain", "unknown"),
                    "answer_type": parent_task.meta.get("answer_type", "text"),
                    "_id": s.get("id", ""),
                }
            )

        correct_tag = _diagnose_dominant_tag(cluster_items)
        permuted_tag = _pick_permuted_tag(correct_tag, self._rng)

        logger.info(
            f"  [permuted-tag] correct_tag={correct_tag} -> permuted_tag={permuted_tag}"
        )

        runner = Runner(self.model_name)

        generated = generate_cluster_batch(
            runner=runner,
            model_cfg=self.model_cfg,
            cluster_items=cluster_items,
            shared_weakness_tag=permuted_tag,
            max_candidates=max_candidates,
            depth=parent_task.depth + 1,
            min_difficulty_increase=self.min_difficulty_increase,
        )

        followups = []
        for item in generated:
            task = self._item_to_task(item, parent_task)
            if task:
                task.weakness_tag = permuted_tag
                task.meta["_correct_tag"] = correct_tag
                task.meta["_permuted_tag"] = permuted_tag
                followups.append(task)
        return followups


def save(data):
    json.dump(data, open(RESULTS_PATH, "w"), indent=2, default=str)


def load_or_init():
    if RESULTS_PATH.exists():
        return json.load(open(RESULTS_PATH))
    return {"meta": {}, "d0_results": {}, "crosseval_12tag": []}


def run_permuted_crosseval(
    bench_name,
    threshold,
    source_display,
    source_model_id,
    task,
    limit,
    data,
    n,
    d0_precomputed,
):
    logger.info(f"  [permuted-tag] {source_display} x {bench_name.upper()} (N={n})")
    cfg = {"temperature": 0.5, "max_tokens": 32000}

    generator = PermutedTagGenerator(
        model_name=GENERATOR_MODEL, model_cfg=cfg, min_difficulty_increase=1.5
    )

    seed_task = EvalTask(task=task, depth=0, weakness_tag=f"{bench_name}_permuted")
    trajectory = Trajectory(
        seed_tasks=[seed_task], generator=generator, threshold=threshold
    )

    samples = d0_precomputed.get("samples", [])
    pre = TaskResult(
        task_id=seed_task.id,
        score=d0_precomputed.get("score", 0.0),
        scores={},
        metadata={},
        parent_id=None,
        depth=0,
        samples=samples,
    )
    trajectory.results[seed_task.id] = pre
    trajectory.expand(
        target_model=source_model_id,
        max_depth=MAX_DEPTH,
        limit_per_task=limit,
        max_candidates=5,
    )

    for tid, tnode in trajectory.nodes.items():
        if tnode.depth == 0:
            continue
        node_scores = {}
        src = trajectory.results.get(tid)
        if src:
            node_scores[source_display] = {
                "score": src.score,
                "passed": src.score >= threshold,
                "is_source": True,
            }
        for model_id, model_display in TARGET_MODELS:
            if model_display in node_scores:
                continue
            try:
                r = tnode.execute(model=model_id)
                node_scores[model_display] = {
                    "score": r.score,
                    "passed": r.score >= threshold,
                    "is_source": False,
                }
            except Exception as e:
                node_scores[model_display] = {
                    "score": None,
                    "passed": None,
                    "is_source": False,
                    "error": str(e),
                }
        node_meta = tnode.meta or {}
        data["crosseval_12tag"].append(
            {
                "task_id": tid,
                "depth": tnode.depth,
                "weakness_tag": node_meta.get("_permuted_tag", "permuted_unknown"),
                "correct_tag": node_meta.get("_correct_tag", "unknown"),
                "benchmark": bench_name,
                "source_model": source_display,
                "generator": GENERATOR_MODEL,
                "conditioning": "permuted_tag",
                "n": n,
                "seed": SEED,
                "scores": node_scores,
            }
        )
        save(data)


def run_permuted_experiment(benchmarks, n=DEFAULT_N):
    d0_all = load_d0_from_run(RUN_PATH, benchmarks)
    if not d0_all:
        logger.error(f"no D0 results found in {RUN_PATH}")
        sys.exit(1)
    data = load_or_init()
    data["meta"] = {
        "conditioning": "permuted_tag",
        "generator": GENERATOR_MODEL,
        "n": n,
        "seed": SEED,
        "benchmarks": benchmarks,
        "run_id": RUN_ID,
        "timestamp": datetime.now().isoformat(),
    }
    data["d0_results"] = {
        b: {
            m: {k: v for k, v in info.items() if k != "samples"}
            for m, info in d0_all[b].items()
        }
        for b in benchmarks
        if b in d0_all
    }
    save(data)

    for bench_name in benchmarks:
        if bench_name not in d0_all:
            continue
        tau = THRESHOLDS[bench_name]
        task, eff_n = load_benchmark(bench_name, n)
        failing = [
            (md, info)
            for md, info in d0_all[bench_name].items()
            if not info.get("passed")
        ]
        if not failing:
            continue
        for md, info in failing:
            mid = info.get("model_id") or MODEL_ID_MAP.get(md, "")
            if not mid:
                continue
            existing = {
                x["source_model"]
                for x in data["crosseval_12tag"]
                if x.get("benchmark") == bench_name
            }
            if md in existing:
                continue
            run_permuted_crosseval(bench_name, tau, md, mid, task, eff_n, data, n, info)

    logger.info("Done")
    save(data)
    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Conditioning ablation: permuted tag")
    p.add_argument(
        "--bench", nargs="+", choices=ALL_BENCHMARKS, default=["gpqa", "math", "bbh"]
    )
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument(
        "--run",
        type=Path,
        default=RUN_PATH,
        help="Run file whose D0 results seed the expansion",
    )
    p.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = p.parse_args()
    RUN_PATH, RESULTS_PATH = args.run, args.out
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Conditioning ablation: diagnosed tag replaced by a random wrong tag")
    print(f"Benchmarks: {args.bench}  N={args.n}")
    run_permuted_experiment(args.bench, args.n)
