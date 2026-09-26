"""One artifact for every number the post quotes: output/analysis.json, from the per-question results.

Reads output/results/<bench>__<arm>.json (one record per question, collected by src/collect.py).
Arms: jev = Jev-Omni through its decision head (A), jevdigits = Jev-Omni's weights through the LM-head
digit readout (D), base = untouched Gemma 4 12B IT through the same readout (B).
Benches: medium, hard (DecisionBench), blink, mmstar, and the prompt variants <bench>-empty / <bench>-look.

Accuracy is reported two ways: per question ("micro") and each benchmark's own headline aggregate
("native"): mean over case logs (states) on DecisionBench, mean over the 14 tasks on BLINK, per question
on MMStar. Paired differences use the same questions for both arms. Bootstrap intervals resample the unit
the metric averages over: states on DecisionBench (so questions from one case log move together),
questions within each task on BLINK, questions on MMStar.

Temperature T is fit per arm by NLL on DecisionBench medium only, then applied unchanged elsewhere, so
every scaled number except medium's is out-of-sample for T.
"""
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

GRID = [round(0.25 + 0.05 * i, 2) for i in range(196)]  # T in [0.25, 10.0]
N_BOOT, SEED = 2000, 0


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
    # 6 decimals so that a later 3-decimal display rounds once, from the full value
    return {"ece": round(ece, 6),
            "nll": round(-sum(math.log(g) for *_, g in out) / len(out), 6),
            "mean_conf": round(sum(c for c, *_ in out) / len(out), 6),
            "wrong_ge_099": sum(1 for c, ok, _ in out if c >= 0.99 and not ok),
            "wrong": sum(1 for _, ok, _ in out if not ok)}


def fit_T(rows):
    return min(GRID, key=lambda T: stats(rows, T)["nll"])


def mcnemar(b, c):
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def key(r):
    return (r["id"], r.get("qkey"))


def group_of(bench, r):
    """The unit a benchmark's headline metric averages over."""
    base = bench.split("-")[0]
    if base in ("medium", "hard"):
        return r["id"]            # a DecisionBench state (case log) holds 1-9 questions
    if base == "blink":
        return r["task"]
    return None                   # MMStar: per question


def accuracy(bench, rows):
    micro = sum(r["correct"] for r in rows) / len(rows)
    groups = defaultdict(list)
    for r in rows:
        groups[group_of(bench, r)].append(r["correct"])
    native = micro if None in groups else sum(sum(v) / len(v) for v in groups.values()) / len(groups)
    return 100 * micro, 100 * native


def paired(bench, A, B):
    """A minus B on the same questions: McNemar on disagreements, bootstrap CIs for micro and native."""
    keys = sorted(A.keys() & B.keys())
    a_only = sum(A[k]["correct"] and not B[k]["correct"] for k in keys)
    b_only = sum(B[k]["correct"] and not A[k]["correct"] for k in keys)
    base = bench.split("-")[0]
    rng = random.Random(SEED)

    def diffs(sample):
        micro = sum(int(A[k]["correct"]) - int(B[k]["correct"]) for k in sample) / len(sample)
        if base == "mmstar":
            return micro, micro
        g = defaultdict(list)
        for k in sample:
            g[group_of(bench, A[k])].append(int(A[k]["correct"]) - int(B[k]["correct"]))
        return micro, sum(sum(v) / len(v) for v in g.values()) / len(g)

    if base in ("medium", "hard"):      # cluster bootstrap over states
        by = defaultdict(list)
        for k in keys:
            by[A[k]["id"]].append(k)
        units = list(by.values())
        draw = lambda: [k for u in (rng.choice(units) for _ in units) for k in u]
    elif base == "blink":               # stratified: resample questions within each task
        by = defaultdict(list)
        for k in keys:
            by[A[k]["task"]].append(k)
        draw = lambda: [rng.choice(v) for v in by.values() for _ in v]
    else:
        draw = lambda: [rng.choice(keys) for _ in keys]
    point = diffs(keys)
    boots = [diffs(draw()) for _ in range(N_BOOT)]

    def ci(i):
        s = sorted(b[i] for b in boots)
        return [round(100 * s[int(0.025 * N_BOOT)], 2), round(100 * s[int(0.975 * N_BOOT) - 1], 2)]

    return {"n": len(keys), "a_only": a_only, "b_only": b_only, "mcnemar_p": round(mcnemar(a_only, b_only), 5),
            "micro_diff_pp": round(100 * point[0], 2), "micro_diff_ci95_pp": ci(0),
            "native_diff_pp": round(100 * point[1], 2), "native_diff_ci95_pp": ci(1)}


def holm(ps):
    order = sorted(range(len(ps)), key=ps.__getitem__)
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def blink_tasks(A, B):
    rows = []
    for t in sorted({r["task"] for r in A.values()}):
        ks = [k for k in A if A[k]["task"] == t]
        ao = sum(A[k]["correct"] and not B[k]["correct"] for k in ks)
        bo = sum(B[k]["correct"] and not A[k]["correct"] for k in ks)
        rows.append({"task": t, "n": len(ks), "chance": round(100 / A[ks[0]]["n_options"], 1),
                     "jev": round(100 * sum(A[k]["correct"] for k in ks) / len(ks), 1),
                     "base": round(100 * sum(B[k]["correct"] for k in ks) / len(ks), 1),
                     "a_only": ao, "b_only": bo, "p": round(mcnemar(ao, bo), 4)})
    for r, p in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = round(p, 4)
    return rows


def position_bias(rows):
    """How often the arm picks option 1 vs how often option 1 is correct, per option count; and the
    accuracy after dividing out the arm's own average distribution over positions (a diagnostic, fitted
    in-sample on the same questions, in the spirit of contextual calibration)."""
    by_n = defaultdict(list)
    for r in rows:
        by_n[r["n_options"]].append(r)
    out, fixed = {}, 0
    for n, rs in sorted(by_n.items()):
        prior = [sum(r["probs"][i] for r in rs) / len(rs) for i in range(n)]
        for r in rs:
            adj = [p / max(q, 1e-12) for p, q in zip(r["probs"], prior)]
            fixed += max(range(n), key=adj.__getitem__) == r["gold"]
        if len(rs) >= 20:
            out[str(n)] = {"n": len(rs), "pred_first": round(sum(r["pred"] == 0 for r in rs) / len(rs), 3),
                           "gold_first": round(sum(r["gold"] == 0 for r in rs) / len(rs), 3)}
    return {"by_n_options": out, "acc_micro_prior_corrected": round(100 * fixed / len(rows), 2)}


def latency(rows):
    ms = sorted(r["ms"] for r in rows)  # the v2 harness runs untimed warm-up forwards first
    tok = sorted(r["n_tokens"] for r in rows if "n_tokens" in r)
    return {"median_ms": round(ms[len(ms) // 2], 1), "p95_ms": round(ms[int(0.95 * (len(ms) - 1))], 1),
            "median_tokens": tok[len(tok) // 2] if tok else None}


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "output" / "results")
    data = {}
    for f in sorted(root.glob("*__*.json")):
        bench, arm = f.stem.split("__")
        data[(bench, arm)] = {key(r): r for r in json.loads(f.read_text())}
    arms = sorted({a for _, a in data})
    T = {a: fit_T(list(data[("medium", a)].values())) for a in arms if ("medium", a) in data}
    out = {"temperature_fit_on": "decisionbench medium (NLL); the medium 'scaled' rows are in-sample",
           "T": T, "n_boot": N_BOOT, "seed": SEED, "benches": {}}
    order = ["medium", "hard", "blink", "mmstar"]
    for bench in order + sorted({b for b, _ in data} - set(order)):
        entry = {}
        for a in arms:
            if (bench, a) not in data:
                continue
            rows = list(data[(bench, a)].values())
            micro, native = accuracy(bench, rows)
            e = {"n": len(rows), "acc_micro": round(micro, 2), "acc_native": round(native, 2),
                 "raw": stats(rows), "ties": sum(bool(r.get("tie")) for r in rows), **latency(rows)}
            if a in T:
                e["scaled"] = {"T": T[a], **stats(rows, T[a])}
            if "valid_mass" in rows[0]:
                vm = sorted(r["valid_mass"] for r in rows)
                e["valid_mass_min"] = round(vm[0], 4)
                e["valid_mass_lt_0.5"] = sum(v < 0.5 for v in vm)
            if bench in order:
                e["position"] = position_bias(rows)
            entry[a] = e
        for a in arms:
            if a != "base" and (bench, a) in data and (bench, "base") in data:
                entry[f"paired_{a}_vs_base"] = paired(bench, data[(bench, a)], data[(bench, "base")])
        if (bench, "jev") in data and (bench, "jevdigits") in data:
            # the head's own effect: the same fine-tuned weights, read through the head vs through digits
            entry["paired_jev_vs_jevdigits"] = paired(bench, data[(bench, "jev")], data[(bench, "jevdigits")])
        if bench == "blink" and ("blink", "jev") in data:
            entry["per_task_jev_vs_base"] = blink_tasks(data[("blink", "jev")], data[("blink", "base")])
        out["benches"][bench] = entry
    print(json.dumps(out, indent=1))
