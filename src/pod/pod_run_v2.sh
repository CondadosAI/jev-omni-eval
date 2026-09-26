#!/usr/bin/env bash
# Full re-run (v2) after the review fixes: BLINK stem regex, MMStar "nan" options, fp32 LM-head readout,
# explicit warm-up, 20-question image-path equivalence check, per-run checks_*.json.
# Expects src/*.py copied to /workspace/bench. Writes raw JSONL to /workspace/bench/v2/.
set -uo pipefail
cd /workspace/bench
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 HF_HOME=/workspace/hf HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_HIGH_PERFORMANCE=1
exec > >(tee -a run_v2.log) 2>&1
echo "== $(date -u) setup"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
lscpu | grep -E "Model name|^CPU\(s\)"; free -g | head -2; grep PRETTY /etc/os-release
command -v uv >/dev/null || pip install --break-system-packages -q uv
uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python "torch==2.14.*" "torchvision==0.29.*"
uv pip install -q --python .venv/bin/python transformers==5.17.0 accelerate huggingface_hub safetensors numpy Pillow opencv-python-headless soundfile librosa pandas pyarrow
.venv/bin/python -c "import torch; assert torch.cuda.is_available()" || { echo '!! CUDA not usable'; exit 1; }
.venv/bin/python -c "import torchvision, transformers.models.gemma4_unified.processing_gemma4_unified" || { echo '!! processor import failed'; exit 1; }
uv pip freeze --python .venv/bin/python > pip-freeze.txt
JD="--base-repo akhilaaa3/Jev-Omni --base-rev 5addda86ddee081a68fb067477ea100c221b8917"
run() { echo "== $(date -u) $*"; .venv/bin/python "$@" || echo "!! failed: $*"; }
run eval_db.py --model jev  --out v2/main
run eval_db.py --model base --out v2/main
run eval_db.py --model base --out v2/jevdigits $JD
for b in blink mmstar; do
  run eval_img.py --bench $b --model jev  --out v2/main
  run eval_img.py --bench $b --model base --out v2/main
  run eval_img.py --bench $b --model base --out v2/jevdigits $JD
  for v in empty look; do
    run eval_img.py --bench $b --model jev  --state-variant $v --out v2/main
    run eval_img.py --bench $b --model base --state-variant $v --out v2/main
  done
done
echo "== $(date -u) DONE"; ls -la v2/main v2/jevdigits; grep -h "" v2/*/checks_*.json | head -80
