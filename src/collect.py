"""Collect raw harness output (JSONL, one record per question) into output/results/<bench>__<arm>.json
and merge the per-run checks_*.json into output/checks.json.

Layout written by src/pod/pod_run_v2.sh:
  <raw>/main/{jev,base}_<bench>[-<variant>].jsonl      arms A (jev) and B (base)
  <raw>/jevdigits/base_<bench>.jsonl                   arm D: Jev-Omni weights through the digit readout
  <raw>/*/checks_*.json
Usage: python src/collect.py <raw_dir> [output_dir]
"""
import json
import sys
from pathlib import Path

raw = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "output"
(out / "results").mkdir(parents=True, exist_ok=True)
written = []
for sub, arm_of in (("main", {"jev": "jev", "base": "base"}), ("jevdigits", {"base": "jevdigits"})):
    for f in sorted((raw / sub).glob("*_*.jsonl")):
        model, bench = f.stem.split("_", 1)
        arm = arm_of[model]
        rows = [json.loads(line) for line in f.open()]
        for r in rows:
            r["model"] = arm  # arm D is written by the harness under its "base" label
        dst = out / "results" / f"{bench}__{arm}.json"
        dst.write_text(json.dumps(rows, separators=(",", ":")))
        written.append((dst.name, len(rows)))
checks = {f"{p.parent.name}/{p.stem}": json.loads(p.read_text()) for p in sorted(raw.glob("*/checks_*.json"))}
(out / "checks.json").write_text(json.dumps(checks, indent=1))
for name, n in written:
    print(f"{name:28s} {n}")
print(f"checks.json: {len(checks)} runs")
