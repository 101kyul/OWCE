"""Cluster size k -> targeting error (beta_j - theta^w_m)^2.

Produces the data for the context-size calibration curve.

What is measured:
  theta^w_m := E[beta_i | i in F^w_m]   mean difficulty of the items model m
                                        failed under weakness w
  theta_hat  = (1/k) sum beta_hat_i     estimated from k cluster items
  epsilon_k  = beta_hat_j - theta_hat   generated probe's difficulty minus the
                                        estimated ability of the source model

  beta_hat comes from the cross-model pass rate:
      beta_hat_i = -logit(p_i),  p_i = (# models passing i) / N_models

  The claim under test is that E[epsilon^2] falls as k grows -- a larger
  cluster lets the generator estimate theta^w_m more accurately -- and that
  the useful range [k_min, k_max] is where E[epsilon^2] stays under tau.

Setting: MATH, the full pool of the source model's D0 failures (no tag
filter, to keep the pool diverse), with k displayed items per generation.

Generated probes are scored with match(). Reads a run file with D0 samples
and writes the per-k measurements to --out.

    python experiments/context_size_calibration.py --run results/results_n300.json \\
        [--out results/context_size.json]
"""

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from inspect_ai.scorer import match
from owce import EvalTask
from owce.generation import generate_cluster_batch
from owce.generators import Runner

GENERATOR_MODEL = "anthropic/claude-opus-4-6"
GENERATOR_CFG = {"temperature": 0.0, "max_tokens": 4096}

BENCHMARK = "math"
# The tag conditions generation; the pool itself is every D0 failure.
WEAKNESS_TAG = "numeric_computation"

# The same six models whose cross-evaluation defines beta_hat at D0.
ALL_MODELS = [
    ("openai/gpt-4o-mini",                                "GPT-4o-mini"),
    ("anthropic/claude-haiku-4-5-20251001",               "Claude-Haiku-4.5"),
    ("together/meta-llama/Llama-3.3-70B-Instruct-Turbo",  "Llama-3.3-70B"),
    ("together/Qwen/Qwen2.5-7B-Instruct-Turbo",           "Qwen2.5-7B"),
    ("openai/gpt-5.4",                                    "GPT-5.4"),
    ("anthropic/claude-sonnet-4-5-20250929",              "Claude-Sonnet-4.5"),
]
N_MODELS = len(ALL_MODELS)
SOURCE_MODEL = "GPT-4o-mini"
THRESHOLD = 0.75   # pass/fail cutoff for MATH

K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
N_TRIALS = 8
N_PROBES = 3
SEED = 42

# Laplace smoothing for logit stability (half a pseudo-count per side)
SMOOTH_A = 0.5
SMOOTH_B = 0.5

DATA_FILE = ROOT / "results" / "results_n300.json"        # a run file with D0 samples; set by --run
RESULTS_FILE = ROOT / "results" / "context_size.json"     # set by --out


def estimate_beta(pass_count: int, n_models: int) -> float:
    """beta_hat = -logit(p), Laplace-smoothed.

    p = (pass_count + 0.5) / (n_models + 1.0)
      beta_hat > 0: hard item, most models fail
      beta_hat < 0: easy item, most models pass
      beta_hat = 0: half the panel passes
    """
    p = (pass_count + SMOOTH_A) / (n_models + SMOOTH_A + SMOOTH_B)
    return -math.log(p / (1.0 - p))


def _score_val(sample):
    """A per-item score as a float: Inspect's C/I/P letters or a number."""
    sc = sample.get("score")
    if isinstance(sc, (int, float)):
        return float(sc)
    return {"C": 1.0, "correct": 1.0, "P": 0.5, "I": 0.0, "incorrect": 0.0}.get(str(sc))


def build_failure_pool() -> list:
    """Every MATH D0 item the source model failed, with its beta_hat."""
    data = json.load(open(DATA_FILE))
    math_data = data["d0_results"]["math"]

    item_pass_count = defaultdict(int)
    item_n_models = defaultdict(int)
    item_sample = {}

    panel = {d for _, d in ALL_MODELS}
    for model_display, mdata in math_data.items():
        if model_display not in panel:
            continue
        for s in mdata.get("samples", []):
            iid = s.get("id", "")
            if not iid:
                continue
            val = _score_val(s)
            if val is None:
                continue
            item_n_models[iid] += 1
            if val >= 1.0:
                item_pass_count[iid] += 1
            if model_display == SOURCE_MODEL and val < 1.0:
                item_sample[iid] = s

    pool = []
    for iid, s in item_sample.items():
        n_pass = item_pass_count.get(iid, 0)
        n_eval = item_n_models.get(iid, N_MODELS)
        beta_i = estimate_beta(n_pass, n_eval)

        q = s.get("input", "")
        if isinstance(q, list):
            q = str(q)

        pool.append({
            "question":     q,
            "gold_answer":  str(s.get("target", "")),
            "answer":       str(s.get("target", "")),
            "output":       str(s.get("output", ""))[:300],
            "domain":       "mathematics",
            "benchmark":    BENCHMARK,
            "weakness_tag": WEAKNESS_TAG,
            "answer_type":  "text",
            "id":           iid,
            "beta_hat":     beta_i,
            "pass_rate_d0": n_pass / n_eval if n_eval > 0 else 0.0,
        })

    betas = [item["beta_hat"] for item in pool]
    print(f"  Pool: {len(pool)} items")
    print(f"  beta_hat range: [{min(betas):.2f}, {max(betas):.2f}]  "
          f"mean={np.mean(betas):.2f}  std={np.std(betas):.2f}")
    return pool


def cross_eval_probe(probe_item: dict):
    """Score one generated probe on the panel and return (scores, beta_j)."""
    task = EvalTask.from_item(item=probe_item, scorer=match(), depth=1)
    scores = {}
    pass_count = 0
    n_evaluated = 0
    for model_id, model_display in ALL_MODELS:
        try:
            result = task.execute(model=model_id)
            passed = result.score >= THRESHOLD
            scores[model_display] = {"score": result.score, "passed": passed}
            if passed:
                pass_count += 1
            n_evaluated += 1
            print(f"        {model_display:<24} {'pass' if passed else 'fail'}")
        except Exception as e:
            print(f"        {model_display:<24} ERR: {str(e)[:50]}")
            scores[model_display] = {"score": 0.0, "passed": False,
                                     "error": str(e)}

    beta_j = estimate_beta(pass_count, n_evaluated) if n_evaluated else 0.0
    return scores, beta_j


def run_ablation():
    rng = random.Random(SEED)

    print("=" * 65)
    print("Cluster size -> targeting error  (beta_j - theta^w_m)^2")
    print(f"Benchmark: {BENCHMARK}  |  Tag: {WEAKNESS_TAG}")
    print(f"k = {K_VALUES}  |  trials/k = {N_TRIALS}  |  probes/trial = {N_PROBES}")
    print("Metric: E[(beta_j - theta_hat)^2], beta = -logit(cross-model pass rate)")
    print(f"temperature = {GENERATOR_CFG['temperature']} (deterministic)")
    print("=" * 65)

    print("\nBuilding the D0 failure pool with beta_hat_i precomputed...")
    pool = build_failure_pool()
    if len(pool) < max(K_VALUES):
        print(f"Pool ({len(pool)}) < max k ({max(K_VALUES)}). Aborting.")
        return

    if RESULTS_FILE.exists():
        results = json.load(open(RESULTS_FILE))
        print(f"Resuming from {RESULTS_FILE}")
    else:
        results = {
            "probes": [],
            "config": {
                "benchmark":   BENCHMARK,
                "tag":         WEAKNESS_TAG,
                "generator":   GENERATOR_MODEL,
                "temperature": GENERATOR_CFG["temperature"],
                "k_values":    K_VALUES,
                "n_trials":    N_TRIALS,
                "n_probes":    N_PROBES,
                "seed":        SEED,
                "metric":      "E[(beta_j_hat - theta_hat)^2]",
                "beta_method": "-logit(pass_rate) with Laplace smoothing",
                "pool_size":   len(pool),
            },
        }

    done = {(p["k"], p["trial"]) for p in results["probes"]}
    runner = Runner(GENERATOR_MODEL)

    for k in K_VALUES:
        print(f"\n{'-' * 60}\n  k = {k}\n{'-' * 60}")

        for trial in range(1, N_TRIALS + 1):
            # Draw before the resume check so that a resumed run uses the same
            # clusters as an uninterrupted one.
            cluster = rng.sample(pool, k)
            if (k, trial) in done:
                print(f"  Trial {trial:2d}/{N_TRIALS}: skip")
                continue

            print(f"\n  Trial {trial:2d}/{N_TRIALS} (k={k})")

            theta_hat = float(np.mean([c["beta_hat"] for c in cluster]))
            beta_cluster = [c["beta_hat"] for c in cluster]

            print(f"    theta_hat = {theta_hat:+.3f}  "
                  f"(cluster beta: [{min(beta_cluster):.2f}, "
                  f"{max(beta_cluster):.2f}])")

            t0 = time.time()
            generated = generate_cluster_batch(
                runner=runner,
                model_cfg=GENERATOR_CFG,
                cluster_items=cluster,
                shared_weakness_tag=WEAKNESS_TAG,
                max_candidates=N_PROBES,
                depth=1,
                min_difficulty_increase=1.5,
                # show all k items, so k's effect is measured in full
                max_display_items=k,
            )
            gen_time = time.time() - t0
            print(f"    Generated {len(generated)} probes  ({gen_time:.1f}s)")

            if not generated:
                results["probes"].append({
                    "k": k, "trial": trial, "n_generated": 0, "probes": [],
                    "theta_hat": theta_hat, "gen_time": gen_time,
                })
                json.dump(results, open(RESULTS_FILE, "w"), indent=2,
                          default=str)
                continue

            trial_probes = []
            for pi, probe in enumerate(generated, 1):
                print(f"\n    Probe {pi}/{len(generated)}  "
                      f"[d_score={probe.get('difficulty_score','?')}]  "
                      f"Q: {probe.get('question','')[:55]}...")

                t1 = time.time()
                scores, beta_j = cross_eval_probe(probe)
                eval_time = time.time() - t1

                epsilon = beta_j - theta_hat
                epsilon_sq = epsilon ** 2

                print(f"    beta_j={beta_j:+.3f}  theta_hat={theta_hat:+.3f}  "
                      f"eps={epsilon:+.3f}  eps^2={epsilon_sq:.3f}")

                trial_probes.append({
                    "probe_idx":        pi,
                    "question_preview": probe.get("question", "")[:200],
                    "gold_answer":      probe.get("gold_answer", ""),
                    "difficulty_score": probe.get("difficulty_score"),
                    "scores":           scores,
                    "beta_j_hat":       beta_j,
                    "theta_hat":        theta_hat,
                    "epsilon":          epsilon,
                    "epsilon_sq":       epsilon_sq,
                    "eval_time":        eval_time,
                })

            results["probes"].append({
                "k":            k,
                "trial":        trial,
                "n_generated":  len(generated),
                "probes":       trial_probes,
                "theta_hat":    theta_hat,
                "beta_cluster": beta_cluster,
                "gen_time":     gen_time,
                "cluster_ids":  [c["id"] for c in cluster],
            })
            done.add((k, trial))
            json.dump(results, open(RESULTS_FILE, "w"), indent=2, default=str)

            eps_vals = [p["epsilon_sq"] for p in trial_probes]
            print(f"\n    Trial {trial} summary: "
                  f"eps^2 = {[round(v, 3) for v in eps_vals]}  "
                  f"mean={np.mean(eps_vals):.3f}")

    print(f"\n{'=' * 65}\nFINAL ANALYSIS\n{'=' * 65}")
    analyze(results)
    json.dump(results, open(RESULTS_FILE, "w"), indent=2, default=str)
    print(f"\nResults saved: {RESULTS_FILE}")


def analyze(results: dict):
    print(f"\n{'-' * 70}")
    print(f"{'k':>3} | {'n':>4} | {'E[eps^2]':>9} | {'E[|eps|]':>8} | "
          f"{'E[eta]':>7} | {'theta_hat':>9}")
    print(f"{'-' * 70}")

    per_k = {}
    for k in K_VALUES:
        eps_sq_vals, eps_vals, theta_vals = [], [], []

        for e in results["probes"]:
            if e["k"] != k:
                continue
            # The summary value is the mean within a trial, then the mean
            # across trials; pooling every probe instead would weight each
            # trial by how many probes it happened to yield.
            trial = [pr["epsilon_sq"] for pr in e.get("probes", [])
                     if pr.get("epsilon_sq") is not None]
            if not trial:
                continue
            eps_sq_vals.append(float(np.mean(trial)))
            for pr in e.get("probes", []):
                if pr.get("epsilon_sq") is not None:
                    eps_vals.append(pr.get("epsilon"))
                    theta_vals.append(pr.get("theta_hat", 0.0))

        if not eps_sq_vals:
            print(f"{k:>3} | (not run)")
            continue

        arr = np.array(eps_sq_vals)
        eps_arr = np.array(eps_vals)
        mean_esq = arr.mean()
        mean_ae = np.abs(eps_arr).mean()
        # Fisher-information proxy: eta = sech^2(eps/2)
        eta_arr = 1.0 / np.cosh(eps_arr / 2) ** 2
        mean_eta = eta_arr.mean()
        mean_th = np.mean(theta_vals)

        per_k[k] = {"mean_esq": mean_esq, "vals": arr, "mean_eta": mean_eta}
        print(f"{k:>3} | {len(arr):>4} | {mean_esq:>9.4f} | {mean_ae:>8.4f} | "
              f"{mean_eta:>7.4f} | {mean_th:>9.3f}")

    if not per_k:
        return

    ks_done = sorted(per_k.keys())
    esq_seq = [per_k[k]["mean_esq"] for k in ks_done]
    mono = all(esq_seq[i] >= esq_seq[i + 1] for i in range(len(esq_seq) - 1))
    print(f"\nE[eps^2] sequence: {[round(float(v), 4) for v in esq_seq]}")
    print(f"monotonically decreasing: {'yes' if mono else 'no'}")

    if len(ks_done) >= 3:
        tau = min(esq_seq) * 1.20
        valid_ks = [k for k in ks_done if per_k[k]["mean_esq"] <= tau]
        if valid_ks:
            print(f"\ntau = {tau:.4f}  (best E[eps^2] x 1.20)")
            print(f"[k_min, k_max] = [{min(valid_ks)}, {max(valid_ks)}]")

    # Is the best k significantly better than k=1?
    if 1 in per_k and len(ks_done) > 1:
        best_k = ks_done[int(np.argmin(esq_seq))]
        if best_k != 1:
            rng_np = np.random.default_rng(SEED)
            a, b = per_k[1]["vals"], per_k[best_k]["vals"]
            diffs = np.array([
                rng_np.choice(b, len(b), True).mean() -
                rng_np.choice(a, len(a), True).mean()
                for _ in range(5000)
            ])
            ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
            sig = "p<0.05" if ci_hi < 0 else "not significant"
            print(f"\nBootstrap E[eps^2](k={best_k}) - E[eps^2](k=1): "
                  f"delta={diffs.mean():+.4f}  "
                  f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]  {sig}")

    results["analysis"] = {
        str(k): {
            "n":        len(per_k[k]["vals"]),
            "mean_esq": float(per_k[k]["mean_esq"]),
            "mean_eta": float(per_k[k]["mean_eta"]),
        }
        for k in ks_done
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DATA_FILE, help="Run file whose D0 failures form the pool")
    parser.add_argument("--out", type=Path, default=RESULTS_FILE)
    args = parser.parse_args()
    DATA_FILE, RESULTS_FILE = args.run, args.out
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    run_ablation()
