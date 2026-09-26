#!/usr/bin/env bash
# Runs on the pod. Setup, then Jev-Omni (with verification) then Gemma 4 base, then score.
set -uo pipefail
cd /workspace/bench
export HF_HOME=/workspace/hf HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_HIGH_PERFORMANCE=1
exec > >(tee -a run.log) 2>&1
echo "== $(date -u) setup"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
command -v uv >/dev/null || pip install --break-system-packages -q uv
uv venv -q --python 3.12 .venv
# torch/torchvision from PyPI (cu130 wheels; needs driver >= 580). The cu128 index has no torch 2.14.
uv pip install --python .venv/bin/python "torch==2.14.*" "torchvision==0.29.*"
uv pip install -q --python .venv/bin/python transformers==5.17.0 accelerate huggingface_hub safetensors numpy Pillow opencv-python-headless soundfile librosa
.venv/bin/python -c "import torch,transformers;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0),'tf',transformers.__version__)"
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" || { echo '!! CUDA not usable, aborting before downloads'; exit 1; }
.venv/bin/pip freeze > pip-freeze.txt 2>/dev/null || uv pip freeze --python .venv/bin/python > pip-freeze.txt
for m in jev base; do
  echo "== $(date -u) model $m"
  .venv/bin/python eval_db.py --model $m --out results || echo "!! $m failed"
done
.venv/bin/python score.py results > results/summary.json
echo "== $(date -u) DONE"; cat results/summary.json
