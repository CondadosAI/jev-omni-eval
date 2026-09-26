"""One artifact for every number the post quotes: analysis.json, computed from the per-question JSONL.

Reads output/results/<bench>__<arm>.json (one record per question, as written by eval_db.py /
eval_img.py). Arms: jev = Jev-Omni through its decision head, jevdigits = Jev-Omni's weights
through the LM-head digit readout, base = untouched Gemma 4 12B IT through the same readout.

Temperature T is fit per arm by NLL on DecisionBench medium only, then applied unchanged to
DecisionBench hard, BLINK and MMStar (so everything but medium is out-of-sample for T).
"""
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

GRID = [round(0.25 + 0.05 * i, 2) for i in range(196)]  # T in [0.25, 10.0]


def load(path):
    return [json.loads(line) for line in Path(path).open()]


def scaled(probs, T):
    lp = [math.log(max(p, 1e-300)) / T for p in probs]
    m = max(lp)
    e = [math.exp(x - m) for x in lp]
    z = sum(e)
    return [x / z for x in e]


def stats(rows, T=1.0):
    out = []
    for r in rows:
        p = scaled(r["probs"], T) if T != 1.0 else r["probs"]
        pred = max(range(len(p)), key=p.__getitem__)
        out.append((p[pred], pred == r["gold"], max(p[r["gold"]], 1e-12)))
    bins = defaultdict(list)
    for c, ok, _ in out:
        bins[min(int(c * 10), 9)].append((c, ok))
    ece = sum(len(b) / len(out) * abs(sum(c for c, _ in b) / len(b) - sum(o for _, o in b) / len(b))
              for b in bins.values())
    return {"ece": round(ece, 4),
            "nll": round(-sum(math.log(g) for *_, g in out) / len(out), 4),
            "mean_conf": round(sum(c for c, *_ in out) / len(out), 4),
            "wrong_ge_099": sum(1 for c, ok, _ in out if c >= 0.99 and not ok),
            "wrong": sum(1 for _, ok, _ in out if not ok)}


def fit_T(rows):
    return min(GRID, key=lambda T: stats(rows, T)["nll"])


def mcnemar(b, c):
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(J, B, macro_key=None, n_boot=2000, seed=0):
    """Accuracy difference jev - base with a bootstrap 95% CI (resampling questions; for state/task-macro
    metrics, resampling within groups keeps the macro definition)."""
    keys = sorted(J.keys() & B.keys())
    jo = sum(J[k]["correct"] and not B[k]["correct"] for k in keys)
    bo = sum(B[k]["correct"] and not J[k]["correct"] for k in keys)
    d = [int(J[k]["correct"]) - int(B[k]["correct"]) for k in keys]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    return {"n": len(keys), "jev_only": jo, "base_only": bo, "mcnemar_p": round(mcnemar(jo, bo), 5),
            "micro_diff_pp": round(100 * sum(d) / len(d), 2),
            "micro_diff_ci95_pp": [round(100 * boots[int(0.025 * n_boot)], 2),
                                   round(100 * boots[int(0.975 * n_boot) - 1], 2)]}


def latency(rows):
    ms = sorted(r["ms"] for r in rows[1:])  # drop the first (warm-up) question of the run
    return {"median_ms_excl_first": round(ms[len(ms) // 2], 1), "p95_ms_excl_first": round(ms[int(0.95 * (len(ms) - 1))], 1)}


def keyed(rows):
    return {(r["id"], r.get("qkey")): r for r in rows}


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "output" / "results")
    data = {}
    for f in sorted(root.glob("*__*.json")):
        bench, arm = f.stem.split("__")
        data[(bench, arm)] = json.loads(f.read_text())
    arms = sorted({a for _, a in data})
    T = {a: fit_T(data[("medium", a)]) for a in arms if ("medium", a) in data}
    out = {"temperature_fit_on": "decisionbench medium (NLL)", "T": T, "benches": {}}
    order = ["medium", "hard", "blink", "mmstar"]
    for bench in order + sorted({b for b, _ in data} - set(order)):
        entry = {}
        for a in arms:
            if (bench, a) not in data:
                continue
            rows = data[(bench, a)]
            e = {"raw": stats(rows), **latency(rows)}
            if a in T:
                e["scaled"] = {"T": T[a], **stats(rows, T[a])}
            if "valid_mass" in rows[0]:
                e["valid_mass_lt_0.5"] = sum(r["valid_mass"] < 0.5 for r in rows)
                e["valid_mass_lt_0.9"] = sum(r["valid_mass"] < 0.9 for r in rows)
            entry[a] = e
        for a in arms:
            if a != "base" and (bench, a) in data and (bench, "base") in data:
                entry[f"paired_{a}_vs_base"] = paired(keyed(data[(bench, a)]), keyed(data[(bench, "base")]))
        out["benches"][bench] = entry
    print(json.dumps(out, indent=1))
