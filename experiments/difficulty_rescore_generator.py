"""Re-elicit the difficulty score of every generated item of a run from the
run's generator, blind to pass rates, on the same 1-10 scale used at
generation time, and correlate it with the panel's pass rates.

    python experiments/difficulty_rescore_generator.py --run results/results_n300.json \\
        [--out results/difficulty_generator.json]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))
from owce.models import generate_with_inspect
from owce.results import generated_items, item_text, spearman


DEFAULT_MODEL = "anthropic/claude-opus-4-6"   # used when the run file names no generator
MODEL_CFG = {"max_tokens": 1024}
BATCH = 1

PROMPT = """You are calibrating the difficulty of LLM evaluation questions.
For each question below, assign a difficulty score on a 1-10 scale for current frontier
language models (1 = trivial, 5 = moderate, 8 = very hard, 10 = at the edge of frontier
capability). Judge ONLY the intrinsic difficulty of the question; do not consider which
model produced it.

Return STRICT JSON: a list of objects {{"id": "<id>", "difficulty_score": <float>}} with
one entry per question, nothing else.

Questions (benchmark context in brackets):
{items}"""


def parse_scores(txt, ids):
    """Scores from a reply that may be the requested JSON list, a single
    object, or just a number when one item was asked for."""
    body = txt.strip()
    try:
        data = json.loads(body)
    except Exception:
        m = re.search(r"\[[\s\S]*\]", body) or re.search(r"\{[\s\S]*\}", body)
        data = None
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None
        if data is None:
            m = re.search(r"difficulty_score\"?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", body) or \
                re.search(r"\b([0-9]+(?:\.[0-9]+)?)\b", body)
            if not m:
                return {}
            data = float(m.group(1))
    if isinstance(data, (int, float)):
        return {ids[0]: float(data)} if len(ids) == 1 else {}
    if isinstance(data, dict):
        data = [data]
    out = {}
    for i, r in enumerate(data):
        if isinstance(r, dict) and r.get("difficulty_score") is not None:
            rid = str(r.get("id", ids[i] if i < len(ids) else ""))
            if rid in ids:
                out[rid] = float(r["difficulty_score"])
        elif isinstance(r, (int, float)) and i < len(ids):
            out[ids[i]] = float(r)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "difficulty_generator.json")
    args = p.parse_args()
    OUT = args.out
    OUT.parent.mkdir(parents=True, exist_ok=True)
    run = json.load(open(args.run))
    nodes = generated_items(run)
    model = run.get("meta", {}).get("generator") or DEFAULT_MODEL

    done = {}
    if OUT.exists():
        prev = json.load(open(OUT))
        done = {r["task_id"]: r for r in prev.get("items", [])}
        print(f"resume: {len(done)} already scored")

    items = []
    for nd in nodes:
        if nd["task_id"] in done:
            items.append(done[nd["task_id"]])
            continue
        items.append({
            "task_id": nd["task_id"], "benchmark": nd["benchmark"],
            "weakness_tag": nd["weakness_tag"], "depth": nd["depth"],
            "source_model": nd["source_model"],
            "probe_input": item_text(nd),
            "panel_pass_rate": sum(1 for s in nd["scores"].values() if s["passed"]) / len(nd["scores"]),
            "others_pass_rate": (lambda o: sum(o) / len(o))(
                [1 if s["passed"] else 0 for s in nd["scores"].values() if not s["is_source"]]),
            "beta_reelicited": None,
        })

    pending = [it for it in items if it["beta_reelicited"] is None]
    print(f"to score: {len(pending)} / {len(items)}")

    for i in range(0, len(pending), BATCH):
        chunk = pending[i:i + BATCH]
        listing = "\n\n".join(
            f'[id={c["task_id"]}] [{c["benchmark"].upper()}] {c["probe_input"][:1500]}'
            for c in chunk)
        for attempt in range(3):
            try:
                txt = generate_with_inspect(model, MODEL_CFG, [PROMPT.format(items=listing)])[0].strip()
                scores = parse_scores(txt, [c["task_id"] for c in chunk])
                if not scores:
                    raise ValueError(f"no score in reply: {txt[:80]!r}")
                for c in chunk:
                    c["beta_reelicited"] = scores.get(c["task_id"])
                break
            except Exception as e:
                print(f"  batch {i//BATCH}: attempt {attempt+1} failed: {e}")
                time.sleep(5 * (attempt + 1))
        json.dump({"items": items}, open(OUT, "w"), indent=2)
        print(f"  batch {i//BATCH + 1}/{(len(pending)+BATCH-1)//BATCH} done")

    scored = [it for it in items if it["beta_reelicited"] is not None]
    fail_rate = [1 - it["panel_pass_rate"] for it in scored]
    betas = [it["beta_reelicited"] for it in scored]

    res = {"n_scored": len(scored),
           "spearman_beta_vs_panel_fail_rate": spearman(betas, fail_rate),
           "by_benchmark": {}, "by_tag": {}, "by_depth": {}}
    for key, field in (("by_benchmark", "benchmark"), ("by_tag", "weakness_tag"), ("by_depth", "depth")):
        for val in sorted({str(it[field]) for it in scored}):
            sub = [it for it in scored if str(it[field]) == val]
            res[key][val] = {
                "n": len(sub),
                "spearman": spearman([it["beta_reelicited"] for it in sub],
                                     [1 - it["panel_pass_rate"] for it in sub]),
                "mean_beta": sum(it["beta_reelicited"] for it in sub) / len(sub),
                "mean_panel_pass": sum(it["panel_pass_rate"] for it in sub) / len(sub),
            }

    json.dump({"summary": res, "items": items}, open(OUT, "w"), indent=2)
    print("\n== summary ==")
    print(json.dumps(res, indent=2)[:3000])


if __name__ == "__main__":
    main()
