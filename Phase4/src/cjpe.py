"""
Phase-4 CJPE data, chunking, metrics and submission writing - kept light:
  * no JSON / text in memory during training: labels + ids come from a small meta.json, tokens from
    memory-mapped .npy files (built once from the caches in WORK/cache)
  * test labels in the released files are NEVER used (test.json carries a label field and dev.json contains
    every test document). "val" = the 994 dev documents that are not test documents.

Normalisation used for the token caches (same for every split): lower-case + whitespace collapsed, because
train is cased modern text and val/test are lower-cased 1950-90 judgments with hard line wraps.
"""
import hashlib
import json
import pickle
import re
import zipfile
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]                 # .../Capstone Project/Phase4
DATA_DIR = Path("C:/kaggle/working/cs781/hf_data/Phase_4_CJPE")
WORK = Path("C:/kaggle/working/cs781/phase4")
SPLITS = ("train", "val", "test")
_WS = re.compile(r"\s+")


def normalise(text):
    return _WS.sub(" ", text.lower()).strip()


def _key(text):
    return hashlib.md5(normalise(text).encode("utf-8")).hexdigest()


def build_splits():
    """Raw JSON -> de-duplicated normalised splits (WORK/cache/splits_v2.pkl; built once, ~1 min)."""
    f = WORK / "cache" / "splits_v2.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    raw = {s: json.load(open(DATA_DIR / p, encoding="utf-8"))
           for s, p in (("train", "Train/train.json"), ("dev", "Train/dev.json"), ("test", "Test/test.json"))}
    test_keys = {_key(r["text"]) for r in raw["test"]}
    out = {s: [] for s in SPLITS}
    for src, dst in (("train", "train"), ("dev", "val")):
        seen = set()
        for r in raw[src]:
            k = _key(r["text"])
            if k in seen or (src == "dev" and k in test_keys):
                continue
            seen.add(k)
            out[dst].append(dict(id=str(r["id"]), text=normalise(r["text"]), label=int(r["label"])))
    out["test"] = [dict(id=str(r["id"]), text=normalise(r["text"]), label=-1) for r in raw["test"]]   # label dropped
    f.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(f, "wb"))
    return out


def meta():
    """{split: {"ids": [...], "labels": [...], "years": [...]}} - small, cached in WORK/cache/meta.json."""
    f = WORK / "cache" / "meta.json"
    if not f.exists():
        sp = build_splits()
        m = {s: dict(ids=[d["id"] for d in sp[s]], labels=[d["label"] for d in sp[s]],
                     years=[int(d["id"].split("_")[0]) if d["id"].split("_")[0].isdigit() else 0 for d in sp[s]])
             for s in SPLITS}
        json.dump(m, open(f, "w", encoding="utf-8"))
    return json.load(open(f, encoding="utf-8"))


class Tokens:
    """Memory-mapped token ids of one split for one tokenizer: tokens[i] -> int32 array of document i."""
    def __init__(self, tok_name, split, tok=None):
        base = WORK / "cache" / f"tokens_{tok_name}_{split}"
        flat_f, off_f = base.with_suffix(".flat.npy"), base.with_suffix(".off.npy")
        if not flat_f.exists():
            npz = base.with_suffix(".npz")
            if npz.exists():
                z = np.load(npz)
                assert list(z["ids"]) == meta()[split]["ids"], f"{npz} out of date"
                flat, off = z["flat"], z["offsets"]
            else:                                   # tokenize once (needs the tokenizer)
                docs = build_splits()[split]
                parts = [np.array(tok.encode_ordinary(d["text"]), dtype=np.int32) for d in docs]
                off = np.zeros(len(parts) + 1, dtype=np.int64); off[1:] = np.cumsum([len(p) for p in parts])
                flat = np.concatenate(parts)
            np.save(flat_f, flat); np.save(off_f, off)
        self.flat = np.load(flat_f, mmap_mode="r")
        self.off = np.load(off_f)

    def __len__(self):
        return len(self.off) - 1

    def __getitem__(self, i):
        return np.asarray(self.flat[self.off[i]:self.off[i + 1]])

    def length(self, i):
        return int(self.off[i + 1] - self.off[i])


def chunks(toks, L, max_chunks, rng=None):
    """Cut a document into L-token chunks. Training (rng given): a random sorted subset of max_chunks chunks.
    Inference: max_chunks evenly spaced chunks (first and last always kept). -> (list of arrays, rel. positions)."""
    n = len(toks)
    if n == 0:
        return [np.zeros(1, np.int32)], np.array([0.5], np.float32)
    starts = list(range(0, max(1, n - L + 1), L))
    if starts[-1] + L < n:
        starts.append(n - L)
    if len(starts) > max_chunks:
        if rng is not None:
            starts = sorted(rng.sample(starts, max_chunks))
        else:
            keep = np.linspace(0, len(starts) - 1, max_chunks).round().astype(int)
            starts = [starts[i] for i in sorted(set(keep.tolist()))]
    cs = [toks[s:s + L] for s in starts]
    pos = np.array([(s + 0.5 * len(c)) / n for s, c in zip(starts, cs)], np.float32)
    return cs, np.clip(pos, 0, 1)


def scores(pred, gold):
    pred, gold = np.asarray(pred).astype(int), np.asarray(gold).astype(int)
    f, p, r = [], [], []
    for c in (0, 1):
        tp = ((pred == c) & (gold == c)).sum(); fp = ((pred == c) & (gold != c)).sum(); fn = ((pred != c) & (gold == c)).sum()
        pc = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * pc * rc / (pc + rc) if pc + rc else 0.0); p.append(pc); r.append(rc)
    return dict(macro_f1=round(float(np.mean(f)), 4), macro_p=round(float(np.mean(p)), 4),
                macro_r=round(float(np.mean(r)), 4), acc=round(float((pred == gold).mean()), 4),
                pred_pos=round(float(pred.mean()), 4))


def best_threshold(p1, gold):
    """Threshold on P(1) maximising val macro-F1 (train is 41% positive, val/test ~50%)."""
    return max((scores(p1 >= t, gold)["macro_f1"], -abs(t - 0.5), float(t)) for t in np.linspace(0.2, 0.8, 61))[2]


def write_submission(pred_by_id, sub_dir, info=None):
    """prediction.json = [{"id", "pred"}] in test.json order, zipped at the archive root."""
    ids = meta()["test"]["ids"]
    pred = [{"id": i, "pred": int(pred_by_id[i])} for i in ids]
    assert len(pred) == 1517 and all(p["pred"] in (0, 1) for p in pred)
    sub_dir = Path(sub_dir); sub_dir.mkdir(parents=True, exist_ok=True)
    pj = sub_dir / "prediction.json"
    json.dump(pred, open(pj, "w", encoding="utf-8"), indent=1)
    z = sub_dir / "submission.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pj, "prediction.json")
    if info is not None:
        json.dump(info, open(sub_dir / "run_info.json", "w", encoding="utf-8"), indent=1, default=str)
    return z
