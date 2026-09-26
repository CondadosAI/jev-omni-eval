"""DecisionBench: Jev-Omni (trained head) vs Gemma 4 12B IT zero-shot (LM-head digit readout).

Both models get the identical prompt built by Jev-Omni's own `_prompt` and chat template.
- jev:  Jev-Omni's official loader; probabilities from its 256-way decision head.
- base: google/gemma-4-12B-it; P(option k) = P(the model writes the digits of k, then stops),
        read from next-token probabilities, renormalised over the valid options.
Writes one JSON line per question, resumable.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch

JEV_REPO, JEV_REV = "akhilaaa3/Jev-Omni", "5addda86ddee081a68fb067477ea100c221b8917"
BASE_REPO, BASE_REV = "google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
DIGITS = "0123456789"


def questions(split_path: Path):
    for line in split_path.open():
        row = json.loads(line)
        qs, ans = json.loads(row["questions"]), json.loads(row["answers"])
        for key, q in qs.items():
            crit = q["criteria"]
            if q["type"] == "score":
                options, gold = list(crit), int(ans[key])
            else:
                keys = list(crit)
                options = [f"{k}: {v}" for k, v in crit.items()]
                gold = keys.index(str(ans[key]).lower() if q["type"] == "noul" else ans[key])
            yield {"id": row["id"], "qkey": key, "type": q["type"], "state": row["state"],
                   "question": q["instructions"], "options": options, "gold": gold}


class Base:
    """Zero-shot readout from the LM head of the untouched instruct model."""

    def __init__(self, path: str, prompt_fn, device="cuda", dtype=torch.bfloat16):
        import transformers
        from transformers import AutoConfig, AutoProcessor
        cfg = AutoConfig.from_pretrained(path)
        self.model = getattr(transformers, cfg.architectures[0]).from_pretrained(
            path, dtype=dtype, device_map=device).eval()
        self.processor = AutoProcessor.from_pretrained(path)
        tok = self.processor.tokenizer
        self.digit_ids = [tok.convert_tokens_to_ids(d) for d in DIGITS]
        self.prompt_fn, self.device = prompt_fn, device
        self.cache_ok = None  # decided on first multi-digit question by comparing to full recompute

    def _inputs(self, state, question, options):
        content = [{"type": "text", "text": self.prompt_fn(state, question, options)}]
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", enable_thinking=False)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _logprobs(self, logits):
        return torch.log_softmax(logits.float(), -1)

    @torch.inference_mode()
    def predict(self, state, question, options):
        inputs = self._inputs(state, question, options)
        n_tok = inputs["input_ids"].shape[1]
        n = len(options)
        out = self.model(**inputs, use_cache=n >= 10, logits_to_keep=1)
        root = self._logprobs(out.logits[0, -1])
        valid = [str(i + 1) for i in range(n)]
        prefixes = {v[:j] for v in valid for j in range(1, len(v) + 1)}
        # Every string that is a valid option AND a proper prefix of another needs P(stop | s);
        # every proper prefix needs its next-digit distribution.
        need = sorted(s for s in prefixes if any(o != s and o.startswith(s) for o in valid))
        dist = {"": root}
        for s in need:
            dist[s] = self._continue(inputs, out.past_key_values if n >= 10 else None, s)
        logp = []
        for v in valid:
            lp, prefix = 0.0, ""
            for ch in v:
                lp += dist[prefix][self.digit_ids[int(ch)]].item()
                prefix += ch
            if prefix in dist:  # could continue with more digits: multiply by P(no further digit)
                p_digit = dist[prefix][self.digit_ids].exp().sum().item()
                lp += math.log(max(1.0 - p_digit, 1e-12))
            logp.append(lp)
        logp_t = torch.tensor(logp, dtype=torch.float64)
        mass = logp_t.exp().sum().item()  # probability the model puts on well-formed valid answers
        probs = torch.softmax(logp_t, 0).tolist()
        return probs, {"n_tokens": n_tok, "valid_mass": mass, "n_extra_forwards": len(need)}

    def _continue(self, inputs, cache, s):
        ext = torch.tensor([[self.digit_ids[int(c)] for c in s]], device=self.device)
        full_ids = torch.cat([inputs["input_ids"], ext], 1)
        full = {**inputs, "input_ids": full_ids,
                "attention_mask": torch.ones_like(full_ids)}
        if self.cache_ok is not False and cache is not None:
            try:
                c = copy.deepcopy(cache)
                o = self.model(input_ids=ext, past_key_values=c, use_cache=True,
                               attention_mask=torch.ones_like(full_ids), logits_to_keep=1)
                lp_cache = self._logprobs(o.logits[0, -1])
                if self.cache_ok is None:  # verify once against a full recompute
                    ref = self._logprobs(self.model(**full, use_cache=False, logits_to_keep=1).logits[0, -1])
                    top = ref.topk(20).indices
                    err = (lp_cache[top].exp() - ref[top].exp()).abs().max().item()
                    self.cache_ok = err < 5e-3
                    print(f"[base] KV-cache continuation check: max |dp| over top-20 = {err:.2e} "
                          f"-> {'use cache' if self.cache_ok else 'full recompute'}", flush=True)
                    if not self.cache_ok:
                        return ref
                return lp_cache
            except Exception as exc:  # noqa: BLE001
                print(f"[base] cache continuation failed ({exc!r}); full recompute from now on", flush=True)
                self.cache_ok = False
        return self._logprobs(self.model(**full, use_cache=False, logits_to_keep=1).logits[0, -1])


DB_REPO, DB_REV = "akhilaaa3/decision-bench", "19334fec40b54b693a63e1ffd91636651d39e847"


def fetch_split(split):
    """DecisionBench is fetched at a pinned revision, never redistributed. Returns a path named <split>.jsonl."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(DB_REPO, f"data/{split}.jsonl", repo_type="dataset", revision=DB_REV))


MODEL_FILES = ["config.json", "generation_config.json", "model.safetensors", "processor_config.json",
               "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]


def fetch(repo, rev, files):
    """Per-file download: the repo-tree API that snapshot_download calls gets rate-limited (429)."""
    from huggingface_hub import hf_hub_download
    paths = [hf_hub_download(repo, f, revision=rev) for f in files]
    return str(Path(paths[0]).parent)


def load(which: str, device: str, base_repo=BASE_REPO, base_rev=BASE_REV):
    code_only = ["jev_omni.py", "verification.json"]
    jev_path = fetch(JEV_REPO, JEV_REV, code_only + (MODEL_FILES + ["decision_config.json", "head.pt"]
                                                     if which == "jev" else []))
    sys.path.insert(0, jev_path)
    import jev_omni  # noqa: E402
    if which == "jev":
        # The official loader re-resolves the repo at `main`; pin it to the revision downloaded above.
        jev_omni.snapshot_download = lambda *_a, **_k: jev_path
        return jev_omni.load_jev_omni(device=device), jev_path
    base_path = fetch(base_repo, base_rev, MODEL_FILES)
    return Base(base_path, jev_omni._prompt, device=device), base_path


def run(which, clf, split: Path, out: Path, limit=None):
    done = set()
    if out.exists():
        done = {(r["id"], r["qkey"]) for r in map(json.loads, out.open())}
    with out.open("a") as fh:
        for i, q in enumerate(questions(split)):
            if limit is not None and i >= limit:
                break
            if (q["id"], q["qkey"]) in done:
                continue
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if which == "jev":
                r = clf.predict(state=q["state"], question=q["question"], options=q["options"])
                probs, extra = [r["probabilities"][o] for o in q["options"]], {}
                if len(set(q["options"])) != len(q["options"]):
                    raise SystemExit(f"duplicate option strings in {q['id']}/{q['qkey']}")
            else:
                probs, extra = clf.predict(q["state"], q["question"], q["options"])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            pred = max(range(len(probs)), key=probs.__getitem__)
            rec = {"model": which, "split": split.stem, "id": q["id"], "qkey": q["qkey"], "type": q["type"],
                   "n_options": len(probs), "gold": q["gold"], "pred": pred, "confidence": probs[pred],
                   "p_gold": probs[q["gold"]], "correct": pred == q["gold"], "ms": round(ms, 2),
                   "probs": probs, **extra}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"[{which}/{split.stem}] {i} correct={rec['correct']} ms={ms:.0f}", flush=True)


def verify_jev(clf, jev_path):
    ref = json.loads((Path(jev_path) / "verification.json").read_text())
    worst = 0.0
    for case, expected in zip(ref["cases"], ref["reference"]):
        got = clf.predict(**case)["probabilities"]
        worst = max(worst, max(abs(got[k] - v) for k, v in expected.items()))
    print(f"[jev] verification worst |dp| = {worst:.4f} (card reference run: {ref['worst_abs_diff']:.4f})", flush=True)
    return worst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["jev", "base"], required=True)
    ap.add_argument("--splits", nargs="+", default=["medium", "hard"])
    ap.add_argument("--data", type=Path, default=None,
                    help="dir with medium.jsonl / hard.jsonl; default: download DecisionBench from the Hub")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--base-repo", default=BASE_REPO, help="override for local smoke tests")
    ap.add_argument("--base-rev", default=BASE_REV)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    clf, path = load(a.model, a.device, a.base_repo, a.base_rev)
    if a.model == "jev":
        w = verify_jev(clf, path)
        (a.out / "jev_verification.json").write_text(json.dumps({"worst_abs_diff": w}))
    for s in a.splits:
        split = a.data / f"{s}.jsonl" if a.data else fetch_split(s)
        run(a.model, clf, split, a.out / f"{a.model}_{s}.jsonl", a.limit)
