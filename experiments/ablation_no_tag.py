"""Conditioning ablation, no tag: the generator prompt carries no weakness-tag
line at all. Reads the D0 results of a run file and cross-evaluates the
generated items on the panel.

    python experiments/ablation_no_tag.py [--bench gpqa math bbh] [--n 300] \\
        [--run results/results_n300.json] [--out results/ablation_no_tag.json]
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset
from owce import EvalTask, Trajectory
from owce.tasks import TaskResult
from owce.generators import ClusterGenerator, Runner, failed_samples, output_text
from owce.generation import (
    _extract_difficulty_score,
    _repair_json_escapes,
    extract_json_array,
    normalize_choice_letter,
)
from owce.results import load_d0_from_run

os.environ.setdefault("INSPECT_LOG_DIR", str(ROOT / "logs" / "inspect"))

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
(ROOT / "logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            ROOT / "logs" / f"ablation_no_tag_{RUN_ID}.log", encoding="utf-8"
        ),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("ablation_no_tag")

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
RESULTS_PATH = ROOT / "results" / "ablation_no_tag.json"  # set by --out


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


def _build_notag_prompt(
    cluster_items,
    parent_difficulty,
    required_difficulty,
    max_candidates,
    max_display_items=5,
):
    """The prompt of the no-tag arm: the cluster prompt of owce.generation
    without the weakness-tag line, the scale-mismatch strategy, the rules
    block and the adversarial_strategy field."""
    display = cluster_items[:max_display_items]
    examples = ""
    for i, item in enumerate(display, 1):
        examples += f"\n**Example {i}:**\n"
        examples += f"Question: {item.get('question','')}\n"
        examples += (
            f"Correct Answer: {item.get('answer', item.get('gold_answer',''))}\n"
        )
        examples += f"Model's Wrong Answer: {item.get('output','')}\n"
    if len(cluster_items) > max_display_items:
        examples += (
            f"\n...(and {len(cluster_items)-max_display_items} more similar failures)\n"
        )

    return f"""IMPORTANT: Output ONLY a JSON array. Do NOT write any explanation, analysis, or preamble. Start your response directly with `[` and end with `]`.

You are an ADVERSARIAL QUESTION DESIGNER. Your goal is to craft questions that will BREAK the target AI model.

**INTELLIGENCE BRIEFING:**
The target model has a SYSTEMATIC FAILURE PATTERN. It failed on ALL {len(cluster_items)} questions below in the EXACT SAME WAY.
[No specific weakness label — analyze the pattern yourself from the examples]

**EVIDENCE (Model's Failures):**
{examples}

**YOUR MISSION:**
Generate {max_candidates} questions that:
1. EXPLOIT the failure pattern you observe in the examples
2. Include DECEPTIVE ELEMENTS that look easy on the surface but contain the exact reasoning trap
3. Design DISTRACTORS that match the model's observed error pattern
4. Make the correct answer COUNTERINTUITIVE

**ADVERSARIAL STRATEGIES (use at least 2 per question):**
- Cognitive trap: Setup that primes the wrong reasoning path, then requires a reversal
- Misleading surface features: Numbers/words that suggest an incorrect but plausible answer
- Multi-step dependency: Each step's correctness depends on the previous (error compounds)
- Edge case exploitation: Boundary conditions where the model's heuristic breaks down
- Distractor alignment: Wrong answer choices match the model's typical error pattern
- Negation/exception: Rules with exceptions that the model is likely to miss

**DIFFICULTY: {required_difficulty:.1f}+ / 10** (parent was {parent_difficulty:.1f})

**OUTPUT FORMAT:**
```json
[
  {{
    "question": "...",
    "answer_type": "multiple_choice|numeric|boolean|short_answer",
    "choices": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "gold_answer": "A",
    "difficulty_score": 8.5,
    "difficulty_justification": "..."
  }}
]
```"""


def _generate_notag_cluster(
    runner, model_cfg, cluster_items, max_candidates, depth, min_difficulty_increase
):
    """generate_cluster_batch with the no-tag prompt. Its checks differ from
    the main validator: a candidate is dropped only when its difficulty score
    is missing or below the gate, the domain is that of the first cluster
    item, and source_item_ids is not recorded."""
    parent_difficulty = 5.0 + depth
    required_difficulty = parent_difficulty + min_difficulty_increase
    prompt = _build_notag_prompt(
        cluster_items, parent_difficulty, required_difficulty, max_candidates
    )
    raw = runner.generate(model_cfg, [prompt])
    text = raw[0] if raw else ""

    payload = extract_json_array(text) or text
    try:
        data = json.loads(payload)
    except Exception:
        try:
            data = json.loads(_repair_json_escapes(payload))
        except Exception:
            return []
    if not isinstance(data, list):
        return []

    verified = []
    for item in data:
        if len(verified) >= max_candidates:
            break
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        if not q:
            continue
        ds = _extract_difficulty_score(item)
        if ds is None or ds < required_difficulty:
            continue
        vi = {
            "question": q,
            "answer_type": str(item.get("answer_type", "")).strip(),
            "domain": (
                cluster_items[0].get("domain", "unknown")
                if cluster_items
                else "unknown"
            ),
            "gold_answer": str(item.get("gold_answer", "")).strip(),
            "difficulty_score": ds,
            "difficulty_justification": str(
                item.get("difficulty_justification", "")
            ).strip(),
            "cluster_size": len(cluster_items),
            "cluster_weakness": "no_tag",
        }
        if vi["answer_type"] == "multiple_choice":
            choices = item.get("choices")
            if not (isinstance(choices, list) and 3 <= len(choices) <= 5):
                continue
            gold = normalize_choice_letter(vi["gold_answer"], len(choices))
            if gold is None:
                continue
            vi["gold_answer"] = gold
            vi["choices"] = [str(c).strip() for c in choices]
        verified.append(vi)
    return verified


@dataclass
class NoTagGenerator(ClusterGenerator):
    """One all-failure cluster per parent, prompted without a weakness-tag line."""

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

        runner = Runner(self.model_name)
        generated = _generate_notag_cluster(
            runner=runner,
            model_cfg=self.model_cfg,
            cluster_items=cluster_items,
            max_candidates=max_candidates,
            depth=parent_task.depth + 1,
            min_difficulty_increase=self.min_difficulty_increase,
        )

        followups = []
        for item in generated:
            task = self._item_to_task(item, parent_task)
            if task:
                task.weakness_tag = "no_tag"
                followups.append(task)
        return followups


def save(data):
    json.dump(data, open(RESULTS_PATH, "w"), indent=2, default=str)


def load_or_init():
    if RESULTS_PATH.exists():
        return json.load(open(RESULTS_PATH))
    return {"meta": {}, "d0_results": {}, "crosseval_12tag": []}


def run_notag_crosseval(
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
    logger.info(f"  [no-tag] {source_display} x {bench_name.upper()} (N={n})")
    cfg = {"temperature": 0.5, "max_tokens": 32000}

    generator = NoTagGenerator(
        model_name=GENERATOR_MODEL, model_cfg=cfg, min_difficulty_increase=1.5
    )

    seed_task = EvalTask(task=task, depth=0, weakness_tag=f"{bench_name}_notag")
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
        data["crosseval_12tag"].append(
            {
                "task_id": tid,
                "depth": tnode.depth,
                "weakness_tag": "no_tag",
                "benchmark": bench_name,
                "source_model": source_display,
                "generator": GENERATOR_MODEL,
                "conditioning": "no_tag",
                "n": n,
                "seed": SEED,
                "scores": node_scores,
            }
        )
        save(data)


def run_notag_experiment(benchmarks, n=DEFAULT_N):
    d0_all = load_d0_from_run(RUN_PATH, benchmarks)
    if not d0_all:
        logger.error(f"no D0 results found in {RUN_PATH}")
        sys.exit(1)
    data = load_or_init()
    data["meta"] = {
        "conditioning": "no_tag",
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
            run_notag_crosseval(bench_name, tau, md, mid, task, eff_n, data, n, info)

    logger.info("Done")
    save(data)
    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Conditioning ablation: no tag")
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
    print("Conditioning ablation: no weakness tag in the prompt")
    print(f"Benchmarks: {args.bench}  N={args.n}")
    run_notag_experiment(args.bench, args.n)
