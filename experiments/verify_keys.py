"""Blind key check: three solvers from outside the generator's family answer
every generated item from its text alone. The output holds each solver's
answers, its agreement rate with the keys, and the number of items whose key
at least two solvers reproduced. Resumable.

    python experiments/verify_keys.py --run results/results_n300.json \\
        [--run results/another_run.json ...] [--out results/verify_keys.json]
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
from owce.results import generated_items, item_text

SOLVERS = ["openai/gpt-5.4", "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
           "together/deepseek-ai/DeepSeek-V4-Pro-0813"]
CHUNK = 10
LET = "ABCDE"
PROMPT = """Solve this evaluation question. Answer with the letter if multiple-choice, or the number if numeric. Give ONLY the final answer.

[Benchmark: {bench}]
{q}"""


def norm_txt(t):
    return re.sub(r"[^a-z0-9]", "", str(t).lower())


def norm_num(s):
    s = str(s).replace(",", "").replace("$", "")
    m = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", s)
    if not m:
        return None
    v = m[-1]
    try:
        if "/" in v:
            a, b = v.split("/")
            return round(float(a) / float(b), 6)
        return round(float(v), 6)
    except Exception:
        return None


def match(ans, gold, choices):
    """Does a solver's answer reproduce the key? A letter must be the key's
    letter, option text must equal the keyed option after normalisation, and
    numbers must be equal."""
    j, g = str(ans).strip(), str(gold).strip()
    if not j:
        return False
    jl = re.match(r"^\(?([A-Ea-e])\)?[\s:.)]*$", j)
    if jl:
        return jl.group(1).upper() == g.upper()
    if choices and len(g) == 1 and g.upper() in LET[:len(choices)]:
        gt = choices[LET.index(g.upper())]
        return norm_txt(j) != "" and norm_txt(j) == norm_txt(gt)
    nj, ng = norm_num(j), norm_num(g)
    if nj is not None and ng is not None:
        return abs(nj - ng) < 1e-6
    return norm_txt(j) == norm_txt(g) and norm_txt(j) != ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, action="append", required=True, help="Run file(s); repeatable")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "verify_keys.json")
    args = p.parse_args()
    OUT = args.out
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nodes = [n for r in args.run for n in generated_items(json.load(open(r)))]
    if not nodes:
        sys.exit("no generated items found in the run file(s)")
    data = json.load(open(OUT)) if OUT.exists() else {"answers": {}}
    for solver in SOLVERS:
        got = data["answers"].setdefault(solver, {})
        pending = [nd for nd in nodes if got.get(nd["task_id"]) is None]
        if not pending:
            print(f"{solver}: complete")
            continue
        print(f"{solver}: {len(pending)} to solve", flush=True)
        for ci in range(0, len(pending), CHUNK):
            chunk = pending[ci:ci + CHUNK]
            prompts = [PROMPT.format(bench=nd["benchmark"].upper(), q=item_text(nd)[:4000]) for nd in chunk]
            outs = generate_with_inspect(solver, {"max_tokens": 6000}, prompts)
            for nd, o in zip(chunk, outs):
                got[nd["task_id"]] = (o or "").strip()[:200] or None
            json.dump(data, open(OUT, "w"), indent=1)
            print(f"  chunk {ci//CHUNK+1}/{(len(pending)+CHUNK-1)//CHUNK}", flush=True)

    by_id = {nd["task_id"]: nd for nd in nodes}
    agree = {}
    summary = {"n_items": len(nodes), "per_solver": {}}
    for solver, ans in data["answers"].items():
        n = ok = 0
        for tid, a in ans.items():
            nd = by_id.get(tid)
            if nd is None or a is None:
                continue
            hit = match(a, nd["probe_target"], nd.get("probe_choices"))
            n += 1
            ok += hit
            agree.setdefault(tid, []).append(hit)
        summary["per_solver"][solver] = {"n": n, "key_agreement": round(ok / n, 4) if n else None}
    summary["n_with_2_solvers"] = sum(1 for v in agree.values() if len(v) >= 2)
    summary["keys_reproduced_by_2_of_3"] = sum(1 for v in agree.values() if sum(v) >= 2)
    data["summary"] = summary
    json.dump(data, open(OUT, "w"), indent=1)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
