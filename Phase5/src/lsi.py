"""
Phase-5 LSI data, chunking, multi-label metrics, thresholds and submission writing (Phase-4 cjpe.py layout):
  * labels + ids come from a small meta.json, tokens from memory-mapped .npy files (built once per tokenizer)
  * splits: train = train.json de-duplicated (first copy of an identical text kept);
            val   = dev.json de-duplicated and WITHOUT the docs whose text also appears in train (81 docs);
            test  = test.json (no labels released).

Normalisation (same for every split): whitespace collapsed, case KEPT (all splits are cased modern text,
unlike CJPE), and the dataset's mask strings mapped to words that are a single Phase-1 BPE token:
  <SECTION> -> Section   <ACT> -> Act   <ENTITY> -> X
("<ENTITY>" alone is 6 BPE tokens and appears ~56 times per document; the median doc is ~550 tokens.)
"""
import hashlib
import json
import pickle
import re
import zipfile
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]                 # .../Capstone Project/Phase5
DATA_DIR = Path("C:/kaggle/working/cs781/hf_data/Phase_5_LSI")
WORK = Path("C:/kaggle/working/cs781/phase5")
SPLITS = ("train", "val", "test")
N_LABELS = 100
_WS = re.compile(r"\s+")
MASKS = {"<SECTION>": "Section", "<ACT>": "Act", "<ENTITY>": "X"}
_MASK_RE = re.compile("|".join(re.escape(k) for k in MASKS))


def normalise(text):
    return _WS.sub(" ", _MASK_RE.sub(lambda m: MASKS[m.group(0)], text)).strip()


def _key(text):
    return hashlib.md5(_WS.sub(" ", text.lower()).strip().encode("utf-8")).hexdigest()


def build_splits():
    """Raw JSON -> de-duplicated normalised splits (WORK/cache/splits.pkl; built once, ~1 min)."""
    f = WORK / "cache" / "splits.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    raw = {s: json.load(open(DATA_DIR / p, encoding="utf-8"))
           for s, p in (("train", "Train/train.json"), ("dev", "Train/dev.json"), ("test", "Test/test.json"))}
    train_keys = {_key(r["text"]) for r in raw["train"]}
    out = {s: [] for s in SPLITS}
    for src, dst in (("train", "train"), ("dev", "val")):
        seen = set()
        for r in raw[src]:
            k = _key(r["text"])
            if k in seen or (src == "dev" and k in train_keys):
                continue
            seen.add(k)
            labels = sorted({int(l) for l in r["labels"]})
            assert labels and all(0 <= l < N_LABELS for l in labels)
            out[dst].append(dict(id=str(r["id"]), text=normalise(r["text"]), labels=labels))
    out["test"] = [dict(id=str(r["id"]), text=normalise(r["text"]), labels=[]) for r in raw["test"]]
    f.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(f, "wb"))
    return out


def meta():
    """{split: {"ids": [...], "labels": [[...], ...]}} - small, cached in WORK/cache/meta.json."""
    f = WORK / "cache" / "meta.json"
    if not f.exists():
        sp = build_splits()
        json.dump({s: dict(ids=[d["id"] for d in sp[s]], labels=[d["labels"] for d in sp[s]]) for s in SPLITS},
                  open(f, "w", encoding="utf-8"))
    return json.load(open(f, encoding="utf-8"))


def label_matrix(split):
    """[n_docs, 100] 0/1 uint8 gold matrix."""
    labs = meta()[split]["labels"]
    y = np.zeros((len(labs), N_LABELS), np.uint8)
    for i, ls in enumerate(labs):
        y[i, ls] = 1
    return y


class Tokens:
    """Memory-mapped token ids of one split for one tokenizer: tokens[i] -> int32 array of document i."""
    def __init__(self, tok_name, split, tok=None):
        base = WORK / "cache" / f"tokens_{tok_name}_{split}"
        flat_f, off_f = base.with_suffix(".flat.npy"), base.with_suffix(".off.npy")
        if not flat_f.exists():                     # tokenize once (needs the tokenizer)
            docs = build_splits()[split]
            parts = [np.array(tok.encode_ordinary(d["text"]), dtype=np.int32) for d in docs]
            off = np.zeros(len(parts) + 1, dtype=np.int64); off[1:] = np.cumsum([len(p) for p in parts])
            np.save(flat_f, np.concatenate(parts)); np.save(off_f, off)
        self.flat = np.load(flat_f, mmap_mode="r")
        self.off = np.load(off_f)
        assert len(self) == len(meta()[split]["ids"]), f"{base} out of date"

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


# ---------------------------------------------------------------- metrics (Codabench: macro-F1 primary)
def per_label_f1(pred, gold):
    """pred, gold [n, L] 0/1 -> (f1, precision, recall) arrays [L]; 0 when undefined."""
    pred, gold = np.asarray(pred, bool), np.asarray(gold, bool)
    tp = (pred & gold).sum(0).astype(float); fp = (pred & ~gold).sum(0); fn = (~pred & gold).sum(0)
    p = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
    r = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=tp + fn > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=p + r > 0)
    return f, p, r


def scores(pred, gold):
    pred, gold = np.asarray(pred, bool), np.asarray(gold, bool)
    f, p, r = per_label_f1(pred, gold)
    tp = (pred & gold).sum(); fp = (pred & ~gold).sum(); fn = (~pred & gold).sum()
    micro = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    return dict(macro_f1=round(float(f.mean()), 4), macro_p=round(float(p.mean()), 4), macro_r=round(float(r.mean()), 4),
                micro_f1=round(float(micro), 4), subset_acc=round(float((pred == gold).all(1).mean()), 4),
                labels_per_doc=round(float(pred.sum(1).mean()), 2))


# ---------------------------------------------------------------- thresholds -> label sets
GRID = np.round(np.arange(0.02, 0.96, 0.01), 2)


def decode(prob, thr, min_labels=1):
    """prob [n, L], thr scalar or [L] -> 0/1 [n, L]; every document gets at least its top label (gold always
    has >= 1 statute)."""
    pred = prob >= np.asarray(thr)[None] if np.ndim(thr) else prob >= thr
    if min_labels:
        empty = ~pred.any(1)
        pred[empty, prob[empty].argmax(1)] = True
    return pred


def global_threshold(prob, gold):
    """One threshold for all labels maximising macro-F1."""
    return float(max(GRID, key=lambda t: (scores(decode(prob, t), gold)["macro_f1"], -abs(t - 0.5))))


def label_thresholds(prob, gold, shrink=0.0, t_global=None):
    """Per-label thresholds maximising each label's F1 on (prob, gold). shrink in [0, 1] pulls them towards the
    global threshold (0 = independent per-label, 1 = global): guards rare labels against overfitting."""
    gold = np.asarray(gold, bool)
    t_global = global_threshold(prob, gold) if t_global is None else t_global
    P = prob[:, :, None] >= GRID[None, None]                  # [n, L, G]
    G = gold[:, :, None]
    tp = (P & G).sum(0); fp = (P & ~G).sum(0); fn = (~P & G).sum(0)
    f1 = 2 * tp / np.maximum(1, 2 * tp + fp + fn)              # [L, G]
    best = GRID[f1.argmax(1)]
    return (1 - shrink) * best + shrink * t_global


def cross_fit(prob, gold, shrink=0.0, seed=0):
    """Honest estimate of per-label thresholding: tune on one half of val, score the other half, both ways.
    Returns the mean macro-F1 of the two held-out halves."""
    idx = np.random.RandomState(seed).permutation(len(prob)); a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
    out = []
    for tr, te in ((a, b), (b, a)):
        out.append(scores(decode(prob[te], label_thresholds(prob[tr], gold[tr], shrink)), gold[te])["macro_f1"])
    return round(float(np.mean(out)), 4)


def threshold_report(prob, gold):
    """Every thresholding strategy on val -> dict (used for model selection and the report)."""
    tg = global_threshold(prob, gold)
    rep = dict(f1_05=scores(decode(prob, 0.5), gold)["macro_f1"], t_global=tg,
               f1_global=scores(decode(prob, tg), gold)["macro_f1"])
    for s in (0.0, 0.5):
        rep[f"f1_label_s{s}_fit"] = scores(decode(prob, label_thresholds(prob, gold, s, tg)), gold)["macro_f1"]
        rep[f"f1_label_s{s}_cv"] = cross_fit(prob, gold, s)
    return rep


def write_submission(pred_by_id, sub_dir, info=None):
    """prediction.json = [{"id", "pred": [ints]}] in test.json order, zipped at the archive root."""
    ids = meta()["test"]["ids"]
    pred = [{"id": i, "pred": sorted({int(l) for l in pred_by_id[i]})} for i in ids]
    assert len(pred) == 13019 and all(p["pred"] and all(0 <= l < N_LABELS for l in p["pred"]) for p in pred)
    sub_dir = Path(sub_dir); sub_dir.mkdir(parents=True, exist_ok=True)
    pj = sub_dir / "prediction.json"
    json.dump(pred, open(pj, "w", encoding="utf-8"))
    z = sub_dir / "submission.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pj, "prediction.json")
    if info is not None:
        json.dump(info, open(sub_dir / "run_info.json", "w", encoding="utf-8"), indent=1, default=str)
    return z


def pred_dict(pred, split="test"):
    """0/1 matrix [n, L] -> {id: [label ids]}."""
    return {i: np.flatnonzero(row).tolist() for i, row in zip(meta()[split]["ids"], pred)}
