"""Reading the run files the pipeline writes, and the rank correlation the
analysis scripts report."""

import json

ITEMS_KEY = "crosseval_12tag"


def generated_items(run: dict) -> list:
    """The generated items of a run."""
    if ITEMS_KEY not in run:
        raise KeyError(f"run file has no '{ITEMS_KEY}' entry; is it a run of run_experiment.py?")
    return run[ITEMS_KEY]


def load_d0_from_run(path, benchmarks) -> dict:
    """D0 results (score, pass flag, per-item samples) of every model in a run
    file written by run_experiment.py, for the requested benchmarks."""
    run = json.load(open(path))
    d0 = {}
    for bench, per in run.get("d0_results", {}).items():
        if bench not in benchmarks:
            continue
        d0[bench] = {md: info for md, info in per.items()
                     if info.get("samples") and not info.get("error")}
    return d0


def item_text(item: dict) -> str:
    """A generated item as a solver or annotator sees it: the question text
    followed by its lettered choices."""
    q = item["probe_input"]
    ch = item.get("probe_choices")
    if ch:
        q += "\n\n" + "\n".join(f"{'ABCDE'[i]}) {c}" for i, c in enumerate(ch))
    return q


def spearman(xs, ys):
    """Spearman rank correlation with average ranks for ties; None when fewer
    than three pairs or either side is constant."""
    n = len(xs)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None
