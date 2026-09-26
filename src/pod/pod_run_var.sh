#!/usr/bin/env bash
# Runs on the pod: setup, then BLINK for Jev-Omni (with image-path equivalence check) and Gemma 4 base.
set -uo pipefail
cd /workspace/bench
export HF_HOME=/workspace/hf HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_HIGH_PERFORMANCE=1
exec > >(tee -a run_var.log) 2>&1
echo "== $(date -u) setup"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
command -v uv >/dev/null || pip install --break-system-packages -q uv
uv venv -q --python 3.12 .venv
# torch/torchvision from PyPI (cu130 wheels; needs driver >= 580). The cu128 index has no torch 2.14.
uv pip install --python .venv/bin/python "torch==2.14.*" "torchvision==0.29.*"
uv pip install -q --python .venv/bin/python transformers==5.17.0 accelerate huggingface_hub safetensors numpy Pillow opencv-python-headless soundfile librosa pandas pyarrow
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" || { echo '!! CUDA not usable, aborting before downloads'; exit 1; }
.venv/bin/python -c "import torchvision, transformers.models.gemma4_unified.processing_gemma4_unified" || { echo '!! processor import failed'; exit 1; }
uv pip freeze --python .venv/bin/python > pip-freeze.txt
for m in jev base; do
  for b in blink mmstar; do
    for v in empty look; do
      echo "== $(date -u) model $m bench $b state $v"
      .venv/bin/python eval_img.py --bench $b --state-variant $v --model $m --out results || echo "!! $m $b $v failed"
    done
  done
done
for s in blink-empty blink-look mmstar-empty mmstar-look; do .venv/bin/python score_img.py results $s > results/summary_$s.json; done
echo "== $(date -u) DONE"; for s in blink-empty blink-look mmstar-empty mmstar-look; do echo "## $s"; python3 -c "import json;d=json.load(open('results/summary_$s.json'));print({k:(v.get('task_macro'),v.get('ece')) if 'task_macro' in v else v for k,v in d.items()})"; done
