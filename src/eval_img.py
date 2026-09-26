"""BLINK (val): Jev-Omni (trained head) vs Gemma 4 12B IT zero-shot, same inputs.

Images go in the user turn before the text, as Jev-Omni's own loader does (one image, or several
like its 16-frame video path). Text = Jev-Omni's `_prompt(state, question, options)`.
Writes one JSON line per question, resumable.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

import eval_db

BLINK_REPO, BLINK_REV = "BLINK-Benchmark/BLINK", "a3666eb249237ba3d5eca8db21176cc47967e040"
TASKS = ("Art_Style Counting Forensic_Detection Functional_Correspondence IQ_Test Jigsaw "
         "Multi-view_Reasoning Object_Localization Relative_Depth Relative_Reflectance "
         "Semantic_Correspondence Spatial_Relation Visual_Correspondence Visual_Similarity").split()


def blink_questions(tasks):
    for task in tasks:
        path = hf_hub_download(BLINK_REPO, f"{task}/val-00000-of-00001.parquet",
                               repo_type="dataset", revision=BLINK_REV)
        for _, r in pd.read_parquet(path).iterrows():
            images = [Image.open(io.BytesIO(r[f"image_{i}"]["bytes"])).convert("RGB")
                      for i in range(1, 5) if r[f"image_{i}"] is not None]
            # The `prompt` field carries the framing multi-image questions need; drop its choice list.
            stem = re.split(r"\n\s*Select from the following choices", r["prompt"])[0].strip()
            letter = r["answer"].strip("() ")
            n = len(images)
            state = "One image is attached." if n == 1 else f"{n} images are attached, in order (first to last)."
            yield {"id": r["idx"], "task": task, "images": images, "state": state, "question": stem,
                   "options": list(r["choices"]), "gold": "ABCD".index(letter)}


MMSTAR_REPO, MMSTAR_REV = "Lin-Chen/MMStar", "bc98d668301da7b14f648724866e57302778ab27"


def parse_mmstar(text):
    """Two source formats: 'Q\nOptions: A: x, B: y, ...' and 'Hint: ...\nQuestion: Q\nChoices:\n(A) x\n(B) y'."""
    m = re.search(r"\n\s*(?:Choices:\s*\n)?\(A\)", text)
    if m:  # '(A) x' lines, with or without a 'Choices:' header
        head, body = text[:m.start()], text[m.start():]
        question = head.split("Question:", 1)[1].strip() if "Question:" in head else head.strip()
        options = [o.strip() for o in re.split(r"\n?\s*\([A-F]\)\s?", body.replace("Choices:", ""))[1:]]
    else:
        head, body = re.split(r"\n\s*Options:\s*", text, maxsplit=1)
        question = head.strip()
        parts = re.split(r"(?:^|,\s*)([A-F]):\s?", body.strip())
        options = [parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)]
    question = question.replace("<image 1>", "").strip()
    return question, options


def mmstar_questions():
    path = hf_hub_download(MMSTAR_REPO, "mmstar.parquet", repo_type="dataset", revision=MMSTAR_REV)
    for _, r in pd.read_parquet(path).iterrows():
        question, options = parse_mmstar(r["question"])
        yield {"id": f"mmstar-{r['index']}", "task": r["category"], "l2": r["l2_category"],
               "images": [Image.open(io.BytesIO(r["image"] if isinstance(r["image"], bytes) else r["image"]["bytes"])).convert("RGB")],
               "state": "One image is attached.", "question": question, "options": options,
               "gold": "ABCDEF".index(r["answer"].strip())}


STATE_VARIANTS = {  # robustness: the default state text is ours, not the model author's
    "empty": lambda n: "",
    "look": lambda n: "Look carefully at the attached image before answering." if n == 1
    else "Look carefully at the attached images before answering.",
}


def make_inputs(processor, images, text, device):
    content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": text}]
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", enable_thinking=False)
    return {k: v.to(device, dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(device)
            for k, v in inputs.items()}


@torch.inference_mode()
def jev_probs(clf, inputs, n):
    """Jev-Omni's predict() body, on inputs we built (its predict() takes at most one media file)."""
    clf._capture.clear()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        clf.model(**inputs, use_cache=False, **clf._extra)
        p = clf.head(clf._capture["hidden"], torch.tensor([n], device=clf.device))[0, :n].softmax(-1)
    return p.float().cpu().tolist()


@torch.inference_mode()
def base_probs(base, inputs, n):
    assert n <= 9, "single-digit readout only"
    logits = base.model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1]
    lp = torch.log_softmax(logits.float(), -1)
    sel = lp[[base.digit_ids[k + 1] for k in range(n)]]
    return torch.softmax(sel.double(), 0).tolist(), sel.exp().sum().item()


def check_jev_equivalence(clf, q, prompt_fn, device):
    """Our input path must reproduce Jev-Omni's own predict() on a single-image question."""
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        q["images"][0].save(f.name)
        ref = clf.predict(state=q["state"], question=q["question"], options=q["options"],
                          media=f.name, modality="image")
    ours = jev_probs(clf, make_inputs(clf.processor, q["images"],
                                      prompt_fn(q["state"], q["question"], q["options"]), device),
                     len(q["options"]))
    diff = max(abs(a - ref["probabilities"][o]) for a, o in zip(ours, q["options"]))
    print(f"[jev] image-path equivalence vs official predict(): max |dp| = {diff:.2e}", flush=True)
    if diff > 1e-3:
        raise SystemExit("our image input path does not match Jev-Omni's predict(); aborting")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["jev", "base"], required=True)
    ap.add_argument("--bench", choices=["blink", "mmstar"], default="blink")
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, help="per task, for smoke tests")
    ap.add_argument("--state-variant", choices=sorted(STATE_VARIANTS))
    ap.add_argument("--base-repo", default=eval_db.BASE_REPO)
    ap.add_argument("--base-rev", default=eval_db.BASE_REV)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    clf, _ = eval_db.load(a.model, a.device, a.base_repo, a.base_rev)
    import jev_omni  # on sys.path after load()
    processor = clf.processor
    suffix = f"-{a.state_variant}" if a.state_variant else ""
    out = a.out / f"{a.model}_{a.bench}{suffix}.jsonl"
    source = blink_questions(a.tasks) if a.bench == "blink" else mmstar_questions()
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()
    if a.model == "jev":
        check_jev_equivalence(clf, next(blink_questions(["Counting"])), jev_omni._prompt, a.device)
    per_task = {}
    with out.open("a") as fh:
        for q in source:
            per_task[q["task"]] = per_task.get(q["task"], 0) + 1
            if a.limit is not None and per_task[q["task"]] > a.limit:
                continue
            if q["id"] in done:
                continue
            if a.state_variant:
                q["state"] = STATE_VARIANTS[a.state_variant](len(q["images"]))
            n = len(q["options"])
            inputs = make_inputs(processor, q["images"], jev_omni._prompt(q["state"], q["question"], q["options"]),
                                 a.device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            extra = {}
            if a.model == "jev":
                probs = jev_probs(clf, inputs, n)
            else:
                probs, extra["valid_mass"] = base_probs(clf, inputs, n)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            pred = max(range(n), key=probs.__getitem__)
            rec = {"model": a.model, "split": a.bench, "l2": q.get("l2"), "id": q["id"], "task": q["task"],
                   "type": q["task"], "n_images": len(q["images"]), "n_options": n,
                   "n_tokens": int(inputs["input_ids"].shape[1]), "gold": q["gold"], "pred": pred,
                   "confidence": probs[pred], "p_gold": probs[q["gold"]], "correct": pred == q["gold"],
                   "ms": round(ms, 2), "probs": probs, **extra}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if per_task[q["task"]] == 1:
                print(f"[{a.model}/{a.bench}] {q['task']} first: n_img={len(q['images'])} "
                      f"tokens={rec['n_tokens']} ms={ms:.0f}", flush=True)


if __name__ == "__main__":
    main()
