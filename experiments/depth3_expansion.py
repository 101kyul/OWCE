"""Expand one level past the pipeline's D_max = 2: the D2 items a source model
still fails are grouped into (source, benchmark, tag) clusters and expanded
with the same generator and cluster prompt at depth 3, then scored on the
panel.

    python experiments/depth3_expansion.py --run results/results_n300.json \\
        [--run results/another_run.json ...] [--out results/depth3.json]
"""
import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from owce.generation import generate_cluster_batch
from owce.generators import Runner
from owce.tasks import EvalTask
from owce.results import ITEMS_KEY, generated_items, item_text


GENERATOR_MODEL = "anthropic/claude-opus-4-6"
GEN_CFG = {"temperature": 0.5, "max_tokens": 4096}

TARGET_MODELS = [
    ("openai/gpt-5.4", "GPT-5.4"),
    ("anthropic/claude-sonnet-4-5-20250929", "Claude-Sonnet-4.5"),
    ("anthropic/claude-haiku-4-5-20251001", "Claude-Haiku-4.5"),
    ("openai/gpt-4o-mini", "GPT-4o-mini"),
    ("together/meta-llama/Llama-3.3-70B-Instruct-Turbo", "Llama-3.3-70B"),
    ("together/Qwen/Qwen2.5-7B-Instruct-Turbo", "Qwen2.5-7B"),
    ("google/gemini-2.5-pro", "Gemini-2.5-Pro"),
]
MAX_CAND_PER_CLUSTER = 2
MIN_EVALUATORS = 5   # a node is kept only with this many scored evaluators


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, action="append", required=True,
                   help="Run file(s) whose D2 items seed the expansion; repeatable")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "depth3.json")
    args = p.parse_args()
    OUT = args.out
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nodes = [n for r in args.run for n in generated_items(json.load(open(r)))]

    d2fail = [n for n in nodes if n["depth"] == 2
              and not next(s for s in n["scores"].values() if s["is_source"])["passed"]]
    clusters = {}
    for n in d2fail:
        clusters.setdefault((n["source_model"], n["benchmark"], n["weakness_tag"]), []).append(n)
    print(f"D2 source-fail seeds: {len(d2fail)} in {len(clusters)} clusters", flush=True)

    data = json.load(open(OUT)) if OUT.exists() else {"meta": {
        "purpose": "depth-3 expansion", "generator": GENERATOR_MODEL,
        "seed_nodes": len(d2fail)}, "done_clusters": [], ITEMS_KEY: []}

    runner = Runner(GENERATOR_MODEL)

    for key, seeds in sorted(clusters.items()):
        ckey = "|".join(key)
        if ckey in data["done_clusters"]:
            print(f"[skip] {ckey} cached", flush=True)
            continue
        source, bench, tag = key
        items = []
        for s in seeds:
            # Run files keep no D2 transcripts, so the prompt gets a stand-in
            # for the source model's wrong answer.
            items.append({"question": item_text(s)[:2500], "gold_answer": str(s["probe_target"]),
                          "output": "(incorrect answer; transcript not retained)",
                          "domain": bench, "_id": s["task_id"]})
        print(f"\n=== {ckey} ({len(items)} seeds) ===", flush=True)
        try:
            gen_items = generate_cluster_batch(
                runner=runner, model_cfg=GEN_CFG, cluster_items=items,
                shared_weakness_tag=tag, max_candidates=MAX_CAND_PER_CLUSTER,
                depth=3, min_difficulty_increase=1.5)
        except Exception as e:
            print(f"  generation error: {e}", flush=True)
            continue

        for gi in gen_items:
            gi["weakness_tag"] = tag
            task = EvalTask.from_item(item=gi, scorer=None, parent_id=None, depth=3)
            node_scores = {}
            for model_id, disp in TARGET_MODELS:
                try:
                    r = task.execute(model=model_id)
                    sc = float(r.score)
                except Exception as e:
                    print(f"    eval {disp} error: {e}", flush=True)
                    sc = None
                if sc is None:
                    continue
                node_scores[disp] = {"score": sc, "passed": sc >= 0.999,
                                     "is_source": disp == source}
            if len(node_scores) < MIN_EVALUATORS or source not in node_scores:
                print("    insufficient evals, dropping node", flush=True)
                continue
            data[ITEMS_KEY].append({
                "task_id": task.id, "depth": 3, "weakness_tag": tag, "benchmark": bench,
                "source_model": source, "generator": GENERATOR_MODEL,
                "parent_ids": [s["task_id"] for s in seeds],
                "probe_input": gi.get("question", ""), "probe_target": gi.get("gold_answer", ""),
                "probe_choices": gi.get("choices"), "difficulty_score": gi.get("difficulty_score"),
                "scores": node_scores})
            src_ok = node_scores[source]["passed"]
            oth = [v["passed"] for k, v in node_scores.items() if k != source]
            print(f"    node: source {'pass' if src_ok else 'FAIL'}, others "
                  f"{sum(oth)}/{len(oth)} pass", flush=True)

        data["done_clusters"].append(ckey)
        json.dump(data, open(OUT, "w"), indent=2, default=str)

    ns = data[ITEMS_KEY]
    if ns:
        deltas = []
        for n in ns:
            src = n["scores"][n["source_model"]]["score"]
            oth = [v["score"] for k, v in n["scores"].items() if k != n["source_model"]]
            deltas.append(sum(oth) / len(oth) - src)
        sp = sum(1 for n in ns if n["scores"][n["source_model"]]["passed"]) / len(ns)
        pp = sum(sum(1 for v in n["scores"].values() if v["passed"]) / len(n["scores"]) for n in ns) / len(ns)
        print(f"\nD3 nodes={len(ns)}  delta={sum(deltas)/len(deltas):+.3f}  "
              f"source_pass={sp:.3f}  panel_pass={pp:.3f}", flush=True)


if __name__ == "__main__":
    main()
