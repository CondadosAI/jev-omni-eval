"""Score BLINK results: per-task accuracy, task-macro average (BLINK's convention), ECE, paired test."""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def ece(rows):
    bins = defaultdict(list)
    for r in rows:
        bins[min(int(r["confidence"] * 10), 9)].append(r)
    return sum(len(b) / len(rows) * abs(sum(x["confidence"] for x in b) / len(b)
                                       - sum(x["correct"] for x in b) / len(b)) for b in bins.values())


def mcnemar(b, c):
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def summarize(rows):
    tasks = defaultdict(list)
    for r in rows:
        tasks[r["task"]].append(r)
    per = {t: round(100 * sum(x["correct"] for x in v) / len(v), 1) for t, v in sorted(tasks.items())}
    single = [r for r in rows if r["n_images"] == 1]
    multi = [r for r in rows if r["n_images"] > 1]
    acc = lambda rs: round(100 * sum(r["correct"] for r in rs) / len(rs), 2) if rs else None
    ms = sorted(r["ms"] for r in rows)
    return {"n": len(rows), "task_macro": round(sum(per.values()) / len(per), 2), "micro": acc(rows),
            "single_image_micro": acc(single), "multi_image_micro": acc(multi),
            "ece": round(ece(rows), 4),
            "confidence_gap_pp": round(100 * (sum(r["confidence"] for r in rows) / len(rows)
                                              - sum(r["correct"] for r in rows) / len(rows)), 2),
            "median_ms": round(ms[len(ms) // 2], 1), "per_task": per}


if __name__ == "__main__":
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    BENCH = sys.argv[2] if len(sys.argv) > 2 else "blink"
    res = {m: summarize([json.loads(l) for l in (d / f"{m}_{BENCH}.jsonl").open()])
           for m in ("jev", "base") if (d / f"{m}_{BENCH}.jsonl").exists()}
    if len(res) == 2:
        J = {json.loads(l)["id"]: json.loads(l) for l in (d / f"jev_{BENCH}.jsonl").open()}
        B = {json.loads(l)["id"]: json.loads(l) for l in (d / f"base_{BENCH}.jsonl").open()}
        common = J.keys() & B.keys()
        jo = sum(J[k]["correct"] and not B[k]["correct"] for k in common)
        bo = sum(B[k]["correct"] and not J[k]["correct"] for k in common)
        res["paired"] = {"n": len(common), "jev_only": jo, "base_only": bo, "mcnemar_p": mcnemar(jo, bo)}
    print(json.dumps(res, indent=1))
