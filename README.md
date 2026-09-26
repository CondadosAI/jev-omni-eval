# jev-omni-eval

Companion code and data for
[What Jev-Omni's decision head buys: 9 points on text, nothing on images](https://condados.ai/blog/jev-omni-decision-head-benchmark)
on condados.ai.

[Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni) answers a multiple-choice question with one
probability per option by reading Gemma 4 12B's last hidden state through a trained 256-way linear
head. This repository reads the same forward pass three ways and compares them on the same questions:

| Arm | Weights | Readout |
|---|---|---|
| `jev` (A) | Jev-Omni (Gemma 4 12B IT + merged rank-512 LoRA) | Jev-Omni's decision head |
| `jevdigits` (D) | Jev-Omni | next-token probability of the option digits `1`…`N` |
| `base` (B) | untouched `google/gemma-4-12B-it` | next-token probability of the option digits |

## Results

Per-question accuracy, paired against the untouched model, with 95% bootstrap intervals
(`output/analysis.json`):

| Benchmark | Questions | A − B | D − B | ECE raw, A / B | ECE after one temperature, A / B |
|---|---:|---|---|---|---|
| DecisionBench medium | 293 | +8.5 [+4.4, +13.0] | +2.4 [−2.4, +7.2] | 0.026 / 0.179 | 0.042 / 0.067 |
| DecisionBench hard | 293 | +4.8 [0.0, +9.6] | −0.3 [−5.5, +4.8] | 0.180 / 0.311 | 0.154 / 0.174 |
| BLINK (val) | 1,901 | −0.8 [−3.0, +1.4] | −10.1 [−12.7, −7.4] | 0.090 / 0.232 | 0.061 / 0.048 |
| MMStar | 1,500 | +0.5 [−1.3, +2.3] | −9.5 [−11.8, −7.1] | 0.140 / 0.275 | 0.108 / 0.095 |

The temperature for each arm (A: 1.2, D: 2.25, B: 3.2) is fitted by negative log-likelihood on
DecisionBench medium only and applied unchanged to the other three benchmarks.

## What is here

| Path | What it is |
|---|---|
| `src/eval_db.py` | DecisionBench harness: builds Jev-Omni's own prompt, runs one forward pass per question, writes one JSON record per question |
| `src/eval_img.py` | BLINK and MMStar harness, same prompt and readouts, images through Gemma 4's processor; checks its image path against Jev-Omni's official `predict()` before running |
| `src/score.py`, `src/score_img.py` | per-run summaries |
| `src/analyze.py` | every number the post quotes, from `output/results/` → `output/analysis.json` |
| `src/pod/*.sh` | the exact scripts run on the RunPod pods (they expect `src/*.py` copied to `/workspace/bench`) |
| `output/results/<bench>__<arm>.json` | one record per question: id, gold option index, per-option probabilities, prediction, latency. No question text and no images |
| `output/analysis.json` | the post's numbers |
| `notebooks/reanalysis.ipynb` | recomputes the tables from `output/results/` with the Python standard library, no GPU |

## Reproduce

The analysis needs nothing but Python 3.10+:

```bash
python src/analyze.py > /tmp/analysis.json && diff /tmp/analysis.json output/analysis.json
```

or open the notebook in Colab:
[notebooks/reanalysis.ipynb](https://colab.research.google.com/github/CondadosAI/jev-omni-eval/blob/main/notebooks/reanalysis.ipynb).

The model runs need a GPU with 48 GB (both checkpoints are 24 GB in bf16). The published numbers
come from RunPod L40S pods with driver 580, torch 2.14.0+cu130 and transformers 5.17.0 (pinned in
`pyproject.toml`):

```bash
uv sync
cd src
uv run python eval_db.py  --model jev  --out ../results     # Jev-Omni head, DecisionBench
uv run python eval_db.py  --model base --out ../results     # untouched Gemma 4
uv run python eval_img.py --bench blink  --model jev --out ../results
uv run python eval_img.py --bench mmstar --model base --out ../results
# arm D: Jev-Omni's weights through the digit readout
uv run python eval_db.py --model base --base-repo akhilaaa3/Jev-Omni \
  --base-rev 5addda86ddee081a68fb067477ea100c221b8917 --out ../results-jevdigits
```

On hosts that expose many CPUs to the container, set `OMP_NUM_THREADS=16`; without it, image
preprocessing ran at about a sixth of the speed on one pod.

## Licence

The code in this repository is Apache-2.0 (see `LICENSE`). **That licence covers the code and the
result files only.** The harness downloads third-party weights and datasets at run time; none of
them is redistributed here, and each keeps its own terms:

| Asset | Revision | Terms |
|---|---|---|
| [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) | `707f0a3b` | Apache-2.0 on the model card, which also links the [Gemma 4 license page](https://ai.google.dev/gemma/docs/gemma_4_license) |
| [akhilaaa3/Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni) | `5addda86` | Apache-2.0 on the model card |
| [DecisionBench](https://huggingface.co/datasets/akhilaaa3/decision-bench) | `19334fec` | Apache-2.0; questions and answers are synthetic, generated with Claude Opus 5 |
| [BLINK](https://huggingface.co/datasets/BLINK-Benchmark/BLINK) (val) | `a3666eb2` | Apache-2.0 on the dataset card |
| [MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar) | `bc98d668` | **No licence is stated on the dataset card.** The harness fetches it for evaluation; nothing from it is included here |

Jev-Omni states that it is not affiliated with, endorsed by, sponsored by, or derived from TypeSafe
AI or its Jev model. Neither is this repository: it measures Jev-Omni, not Jev.
