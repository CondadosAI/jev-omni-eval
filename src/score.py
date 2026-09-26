"""Score result JSONL files with DecisionBench's own definitions (state-macro accuracy, 10-bin ECE)."""
import json
import sys
from collections import defaultdict
from pathlib import Path


def score(rows):
    by_state = defaultdict(list)
    for r in rows:
        by_state[r["id"]].append(r["correct"])
    macro = sum(sum(v) / len(v) for v in by_state.values()) / len(by_state)
    micro = sum(r["correct"] for r in rows) / len(rows)
    bins = defaultdict(list)
    for r in rows:
        bins[min(int(r["confidence"] * 10), 9)].append(r)
    ece = sum(len(b) / len(rows) * abs(sum(x["confidence"] for x in b) / len(b)
                                       - sum(x["correct"] for x in b) / len(b)) for b in bins.values())
    conf = sum(r["confidence"] for r in rows) / len(rows)
    by_type = {}
    for t in ("choice", "noul", "score"):
        tr = [r for r in rows if r["type"] == t]
        by_type[t] = round(100 * sum(r["correct"] for r in tr) / len(tr), 1)
    ms = sorted(r["ms"] for r in rows)
    out = {"questions_scored": len(rows), "correct": sum(r["correct"] for r in rows),
           "accuracy": round(100 * macro, 2), "micro_accuracy": round(100 * micro, 2),
           "states_scored": len(by_state), "ece": round(ece, 4), "mean_confidence": round(conf, 4),
           "confidence_gap_pp": round(100 * (conf - micro), 2), "by_type": by_type,
           "median_ms": round(ms[len(ms) // 2], 1), "p95_ms": round(ms[int(0.95 * (len(ms) - 1))], 1)}
    if "valid_mass" in rows[0]:
        vm = sorted(r["valid_mass"] for r in rows)
        out["median_valid_mass"] = round(vm[len(vm) // 2], 4)
        out["min_valid_mass"] = round(vm[0], 4)
    return out


if __name__ == "__main__":
    res = {}
    for f in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "results").glob("*_*.jsonl")):
        rows = [json.loads(line) for line in f.open()]
        model, split = f.stem.split("_", 1)
        res.setdefault(split, {})[model] = score(rows)
    print(json.dumps(res, indent=1))
