"""
Phase-3 Rhetorical Role data: loading the CL/IT splits, sentence tokenization (cached per tokenizer),
and the two ways a document is fed to the backbone:

  sentence mode  every sentence is its own row                      -> no context inside the backbone
  window mode    consecutive sentences packed into one backbone window, each preceded by a newline
                 separator; a sentence is pooled over its own tokens -> left (causal) or two-sided context

Both modes return rows of token ids + one (row, start, end) span per sentence for pooling.
"""
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]                 # .../Capstone Project/Phase3
DATA_DIR = Path("C:/kaggle/working/cs781/hf_data/Phase_3_RR")
WORK = Path("C:/kaggle/working/cs781/phase3")
DOMAINS = ("CL", "IT")

_vocab = json.load(open(DATA_DIR / "Train" / "label_vocab.json", encoding="utf-8"))
LABELS = [l for l, _ in sorted(_vocab.items(), key=lambda kv: kv[1])]
L2I = {l: i for i, l in enumerate(LABELS)}
N_LABELS = len(LABELS)


@dataclass
class RRDoc:
    id: str
    domain: int                                     # index into DOMAINS
    sents: list
    labels: np.ndarray = None                       # [n] label ids (None for test)
    toks: list = field(default=None, repr=False)    # per-sentence token id arrays (untruncated)

    def __len__(self):
        return len(self.sents)


def load_rr():
    """-> dict(train=[RRDoc], dev=[RRDoc], test=[RRDoc]) over both domains (domain kept per doc)."""
    out = {"train": [], "dev": [], "test": []}
    for di, dom in enumerate(DOMAINS):
        for split in ("train", "dev"):
            for d in json.load(open(DATA_DIR / "Train" / f"{dom}_{split}.json", encoding="utf-8")):
                assert len(d["text"]) == len(d["labels"]), d["id"]
                out[split].append(RRDoc(d["id"], di, d["text"], np.array([L2I[l] for l in d["labels"]])))
        for k, sents in json.load(open(DATA_DIR / "Test" / f"{dom}_test.json", encoding="utf-8")).items():
            out["test"].append(RRDoc(k, di, sents))
    return out


def tokenize_docs(docs, tok, tok_name):
    """Fills doc.toks (one int32 array per sentence). Cached per tokenizer in WORK/cache."""
    cache_f = WORK / "cache" / f"sent_tokens_{tok_name}.pkl"
    cache = pickle.load(open(cache_f, "rb")) if cache_f.exists() else {}
    new = 0
    for d in docs:
        if d.id not in cache:
            cache[d.id] = [np.array(tok.encode_ordinary(s.strip()), dtype=np.int32) for s in d.sents]
            new += 1
        d.toks = cache[d.id]
    if new:
        cache_f.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(cache, open(cache_f, "wb"))
    return docs


def _truncate(t, n):
    return t if len(t) <= n else t[:n]


def sentence_rows(toks, sep, max_sent_tok):
    """Sentence mode: one row per sentence (an empty sentence becomes the separator alone)."""
    rows = [_truncate(t, max_sent_tok) if len(t) else sep for t in toks]
    spans = np.array([(i, 0, len(r)) for i, r in enumerate(rows)], dtype=np.int64).reshape(-1, 3)
    return rows, spans


def pack_windows(lens, L, overlap=0.0):
    """Greedy packing of consecutive pieces (sentence + separator) into windows of at most L tokens.
    Consecutive windows share about `overlap` of their tokens (whole sentences). -> list of (a, b)."""
    n, out, i = len(lens), [], 0
    while i < n:
        j, tot = i, 0
        while j < n and tot + lens[j] <= L:
            tot += lens[j]; j += 1
        j = max(j, i + 1)
        out.append((i, j))
        if j >= n:
            break
        if overlap <= 0:
            i = j
        else:                                       # step back so that ~overlap of the tokens repeat
            k, back = j, 0
            while k - 1 > i and back + lens[k - 1] <= overlap * tot:
                k -= 1; back += lens[k]
            i = k
    return out


def window_rows(toks, sep, L, max_sent_tok, overlap=0.0, causal=True):
    """Window mode. Every sentence is [sep] + tokens; its span covers the tokens only. When windows overlap,
    a sentence keeps the copy with the most context: the most left context for causal backbones, the
    most central position for bidirectional ones."""
    pieces = [np.concatenate([sep, _truncate(t, max_sent_tok)]).astype(np.int32) for t in toks]
    lens = [len(p) for p in pieces]
    rows, spans, best = [], np.zeros((len(toks), 3), dtype=np.int64), np.full(len(toks), -1, dtype=np.int64)
    for a, b in pack_windows(lens, L, overlap):
        r = len(rows)
        rows.append(np.concatenate(pieces[a:b]))
        pos, total = 0, len(rows[-1])
        for s in range(a, b):
            st, en = pos + len(sep), pos + lens[s]
            if en == st:                            # empty sentence -> pool its separator
                st = pos
            score = st if causal else min(st, total - en)
            if score > best[s]:
                best[s] = score; spans[s] = (r, st, en)
            pos += lens[s]
    return rows, spans


def label_transitions(docs, smooth=1.0):
    """log P(label_t | label_{t-1}) estimated on training docs (CRF init / Viterbi prior)."""
    c = np.full((N_LABELS, N_LABELS), smooth)
    for d in docs:
        np.add.at(c, (d.labels[:-1], d.labels[1:]), 1)
    return np.log(c / c.sum(1, keepdims=True))


def label_counts(docs):
    return np.bincount(np.concatenate([d.labels for d in docs]), minlength=N_LABELS)
