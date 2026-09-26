# jev-omni-eval

Companion code and data for
[Jev-Omni, measured three ways](https://condados.ai/blog/jev-omni-decision-head-benchmark) on condados.ai.

[Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni) answers a multiple-choice question with one
probability per option by reading Gemma 4 12B's last hidden state through a trained 256-way linear
head, on top of weights fine-tuned with two merged LoRAs. This repository reads the same kind of input
three ways and compares them on the same questions:

| Readout | Weights | How the probability is read |
|---|---|---|
| **A** `jev` | Jev-Omni | Jev-Omni's decision head (its official loader) |
| **B** `base` | untouched `google/gemma-4-12B-it` | next-token probability of the option digits `1`…`N` |
| **C** `jevdigits` | Jev-Omni | the same digit readout as B, ignoring the head |

## Results

Per-question accuracy difference against B with a 95% bootstrap interval (case logs resampled on
DecisionBench, questions within tasks on BLINK), and expected calibration error before and after one
temperature per readout (fitted on DecisionBench medium, so the medium "scaled" column is in-sample).
The post's headline accuracies use each benchmark's own aggregate (per case log on DecisionBench, per
task on BLINK); both aggregates are in `output/analysis.json`.

| Benchmark | Questions | A − B | C − B | ECE A / B / C | ECE after one temperature A / B / C |
|---|---:|---|---|---|---|
| DecisionBench medium | 293 | +8.5 [+4.5, +12.9] | +2.4 [−2.3, +7.2] | 0.026 / 0.179 / 0.130 | 0.042 / 0.071 / 0.082 |
| DecisionBench hard | 293 | +4.4 [+0.4, +8.3] | −0.3 [−4.8, +4.1] | 0.180 / 0.305 / 0.227 | 0.154 / 0.175 / 0.131 |
| BLINK (val) | 1,901 | −0.5 [−2.7, +1.6] | −9.6 [−12.3, −6.9] | 0.093 / 0.226 / 0.387 | 0.064 / 0.039 / 0.233 |
| MMStar | 1,498 | +0.6 [−1.3, +2.5] | −9.2 [−11.6, −6.7] | 0.142 / 0.279 / 0.355 | 0.109 / 0.100 / 0.212 |

Temperatures: A 1.2, B 3.2, C 2.25. MMStar excludes the two questions whose keyed answer is the literal option "nan" (1,498 of 1,500).

## What is here

| Path | What it is |
|---|---|
| `src/eval_db.py` | DecisionBench harness: Jev-Omni's own prompt, one forward pass per question, one JSON line per question |
| `src/eval_img.py` | BLINK and MMStar harness, same prompt and readouts; checks its image path against Jev-Omni's official `predict()` on 20 questions before running |
| `src/collect.py` | raw JSONL from a run → `output/results/<bench>__<arm>.json` and `output/checks.json` |
| `src/analyze.py` | every number the post quotes, from `output/results/` → `output/analysis.json` |
| `src/score.py`, `src/score_img.py` | quick per-run summaries while a run is going |
| `src/pod/pod_run_v2.sh` | the script that produced the published results on a RunPod L40S (expects `src/*.py` copied to `/workspace/bench`) |
| `output/results/` | one record per question: id, gold option index, per-option probabilities, prediction, latency, token count. No question text and no images |
| `output/checks.json` | the harness self-checks: Jev-Omni's published verification cases, the image-path equivalence check, the fp32 output-layer check, the KV-cache check, the skipped MMStar items |
| `output/analysis.json` | the post's numbers |
| `notebooks/reanalysis.ipynb` | recomputes the tables from `output/results/` and checks them against `output/analysis.json`, no GPU |

## Reproduce

The analysis needs only the Python standard library (plain `python`, not `uv run`, which would install
torch):

```bash
python src/analyze.py > /tmp/analysis.json && diff /tmp/analysis.json output/analysis.json
```

or open the notebook in Colab:
[notebooks/reanalysis.ipynb](https://colab.research.google.com/github/CondadosAI/jev-omni-eval/blob/main/notebooks/reanalysis.ipynb).

The model runs need a GPU with 48 GB (each checkpoint is 24 GB in bf16) and the environment pinned in
`pyproject.toml` (torch 2.14.0 with CUDA 13 wheels, so an NVIDIA driver ≥ 580; transformers 5.17.0).
`src/pod/pod_run_v2.sh` runs everything; the individual steps are:

```bash
uv sync && cd src
JD="--base-repo akhilaaa3/Jev-Omni --base-rev 5addda86ddee081a68fb067477ea100c221b8917"
uv run python eval_db.py  --model jev  --out ../raw/main              # A on DecisionBench
uv run python eval_db.py  --model base --out ../raw/main              # B
uv run python eval_db.py  --model base --out ../raw/jevdigits $JD     # C
uv run python eval_img.py --bench blink --model jev  --out ../raw/main   # and base, and C as above
uv run python eval_img.py --bench blink --model jev  --state-variant empty --out ../raw/main
cd .. && python src/collect.py raw && python src/analyze.py > output/analysis.json
```

On hosts that expose many CPUs to the container, set `OMP_NUM_THREADS=16`; one pod ran image
preprocessing at about a sixth of the speed without it.

## Licence

The code in this repository is Apache-2.0 (see `LICENSE`). **That licence covers the code and the
result files only.** The harness downloads third-party weights and datasets at run time; none of them
is redistributed here, and each keeps its own terms:

| Asset | Revision | Terms |
|---|---|---|
| [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) | `707f0a3b` | Apache-2.0 on the model card, which also links the [Gemma 4 license page](https://ai.google.dev/gemma/docs/gemma_4_license) |
| [akhilaaa3/Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni) | `5addda86` | Apache-2.0 on the model card |
| [DecisionBench](https://huggingface.co/datasets/akhilaaa3/decision-bench) | `19334fec` | Apache-2.0; questions and answers are synthetic, generated with Claude Opus 5 |
| [BLINK](https://huggingface.co/datasets/BLINK-Benchmark/BLINK) (val) | `a3666eb2` | Apache-2.0 on the dataset card, which covers the annotations; the images come from existing datasets and web search with their own terms (the relative-depth photos are Flickr images from Depth in the Wild) |
| [MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar) | `bc98d668` | **No licence is stated on the dataset card.** Its images come from other benchmarks (SEED-Bench, MMBench, MathVista, AI2D, MMMU, ScienceQA) with their own terms |

`output/cover-bg.png`, the background of the post's cover, contains an AI2D diagram (MMStar question 1285) by the Allen Institute for AI under CC BY-SA 4.0; that image keeps its licence.

Jev-Omni states that it is not affiliated with, endorsed by, sponsored by, or derived from TypeSafe AI
or its Jev model. Neither is this repository: it measures Jev-Omni, not Jev.
