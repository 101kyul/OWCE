"""Difficulty scores for every generated item of a run from annotator models
outside the generator's family, blind to pass rates, on the 1-10 scale the
generator used. The generator's own scores (difficulty_rescore_generator.py)
are merged in when given. Reports per-annotator, ensemble and inter-annotator
Spearman correlations and the mean difficulty at D1 and D2.

    python experiments/difficulty_rescore_panel.py --run results/results_n300.json \\
        [--generator-scores results/difficulty_generator.json] [--out results/difficulty_panel.json]
"""
import argparse
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from owce.models import generate_with_inspect
from owce.results import generated_items, item_text, spearman


ANNOTATORS = ["openai/gpt-5.4", "google/gemini-2.5-pro"]
CHUNK = 10

# The generator's 1-10 scale, anchored at frontier capability; no pass-rate information.
PROMPT = """You are calibrating the difficulty of an LLM evaluation question.
Assign a difficulty score on a 1-10 scale for current frontier language models
(1 = trivial, 5 = moderate, 8 = very hard, 10 = at the edge of frontier capability).
Judge ONLY the intrinsic difficulty of the question itself.

Return STRICT JSON only: {{"difficulty_score": <float>}}

[Benchmark: {bench}]
Question:
{question}"""


def parse_score(txt):
    try:
        m = re.search(r'"difficulty_score"\s*:\s*([0-9.]+)', txt)
        if m:
            return float(m.group(1))
        m = re.search(r"\b([0-9](?:\.[0-9])?|10)\s*/\s*10", txt)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--generator-scores", type=Path, default=None,
                   help="Output of difficulty_rescore_generator.py, merged as a third annotator")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "difficulty_panel.json")
    args = p.parse_args()
    OUT = args.out
    OUT.parent.mkdir(parents=True, exist_ok=True)
    run = json.load(open(args.run))
    nodes = generated_items(run)

    data = {"annotations": {}}
    if OUT.exists():
        data = json.load(open(OUT))

    if args.generator_scores and args.generator_scores.exists():
        generator = run.get("meta", {}).get("generator", "generator")
        generator_items = json.load(open(args.generator_scores))["items"]
        data["annotations"][generator] = {
            it["task_id"]: it["beta_reelicited"] for it in generator_items
            if it.get("beta_reelicited") is not None}

    for model in ANNOTATORS:
        got = data["annotations"].get(model, {})
        pending = [nd for nd in nodes if nd["task_id"] not in got or got[nd["task_id"]] is None]
        if not pending:
            print(f"{model}: complete ({len(got)})")
            continue
        print(f"{model}: scoring {len(pending)} probes")
        for ci in range(0, len(pending), CHUNK):
            chunk = pending[ci:ci + CHUNK]
            prompts = [PROMPT.format(bench=nd["benchmark"].upper(), question=item_text(nd)[:3000])
                       for nd in chunk]
            outputs = generate_with_inspect(model, {"max_tokens": 30000}, prompts)
            for nd, out in zip(chunk, outputs):
                got[nd["task_id"]] = parse_score(out or "")
            data["annotations"][model] = got
            json.dump(data, open(OUT, "w"), indent=2)
            print(f"  chunk {ci//CHUNK+1}/{(len(pending)+CHUNK-1)//CHUNK} saved")

    fail = {}
    depth = {}
    for nd in nodes:
        fail[nd["task_id"]] = 1 - sum(1 for s in nd["scores"].values() if s["passed"]) / len(nd["scores"])
        depth[nd["task_id"]] = nd["depth"]

    summary = {"per_annotator": {}, "ensemble": None, "inter_annotator": {}}
    ann = data["annotations"]
    for model, scores in ann.items():
        ids = [t for t, v in scores.items() if v is not None and t in fail]
        summary["per_annotator"][model] = {
            "n": len(ids),
            "spearman_vs_fail": spearman([scores[t] for t in ids], [fail[t] for t in ids]),
            "mean_beta_d1": (lambda s: sum(s) / len(s) if s else None)([scores[t] for t in ids if depth[t] == 1]),
            "mean_beta_d2": (lambda s: sum(s) / len(s) if s else None)([scores[t] for t in ids if depth[t] == 2]),
        }
    models = list(ann.keys())
    # Ensemble: the mean score of every item at least two annotators scored
    scored = {t: [ann[m][t] for m in models if ann[m].get(t) is not None] for t in fail}
    common = [t for t, v in scored.items() if len(v) >= 2]
    if len(common) >= 3 and len(models) >= 2:
        ens = [sum(scored[t]) / len(scored[t]) for t in common]
        summary["ensemble"] = {"n": len(common),
                               "spearman_vs_fail": spearman(ens, [fail[t] for t in common])}
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                pair = [t for t in fail if ann[models[i]].get(t) is not None and ann[models[j]].get(t) is not None]
                summary["inter_annotator"][f"{models[i]} vs {models[j]}"] = {
                    "n": len(pair),
                    "spearman": spearman([ann[models[i]][t] for t in pair], [ann[models[j]][t] for t in pair])}

    data["summary"] = summary
    json.dump(data, open(OUT, "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
