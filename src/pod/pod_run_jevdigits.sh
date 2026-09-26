#!/usr/bin/env bash
# Arm D: Jev-Omni's fine-tuned backbone read through the LM-head digit logits (no decision head).
set -uo pipefail
cd /workspace/bench
export HF_HOME=/workspace/hf HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_HIGH_PERFORMANCE=1
while ! grep -q "== .* DONE" run_var.log 2>/dev/null; do sleep 20; done
exec > >(tee -a run_jevdigits.log) 2>&1
R="--base-repo akhilaaa3/Jev-Omni --base-rev 5addda86ddee081a68fb067477ea100c221b8917 --out results-jevdigits"
echo "== $(date -u) jevdigits decisionbench"; .venv/bin/python eval_db.py --model base $R || echo "!! db failed"
for b in blink mmstar; do echo "== $(date -u) jevdigits $b"; .venv/bin/python eval_img.py --bench $b --model base $R || echo "!! $b failed"; done
.venv/bin/python score.py results-jevdigits > results-jevdigits/summary_db.json
for b in blink mmstar; do .venv/bin/python score_img.py results-jevdigits $b > results-jevdigits/summary_$b.json; done
echo "== $(date -u) DONE"; cat results-jevdigits/summary_*.json
