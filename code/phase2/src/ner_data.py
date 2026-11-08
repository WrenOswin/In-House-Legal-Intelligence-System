"""
Phase-2 data: documents, gold spans, BIO token labels, sliding windows, and the way back
from token predictions to character spans.
"""
import json
from pathlib import Path

import numpy as np

from bpe import encode_with_offsets

DATA_DIR = Path("C:/kaggle/working/cs781/hf_data/Phase_2_LNER")
PROJECT_DIR = Path(__file__).resolve().parents[1]                 # .../Capstone Project/Phase2
SPLIT_FILE = PROJECT_DIR / "data" / "val_split.json"

LABELS = (DATA_DIR / "Train" / "labels.txt").read_text(encoding="utf-8").split()
TAGS = ["O"] + [f"{p}-{l}" for l in LABELS for p in ("B", "I")]   # 25 tags
TAG2ID = {t: i for i, t in enumerate(TAGS)}
N_TAGS = len(TAGS)
IGNORE = -100


def b_id(label): return TAG2ID["B-" + label]
def i_id(label): return TAG2ID["I-" + label]


# ----------------------------------------------------------------------------- loading
def load_split():
    """Returns texts {id: str}, gold {id: [[s,e,label], ...]}, test texts, split {train: [...], val: [...]}."""
    texts, gold = {}, {}
    for f in ("fold1", "fold2"):
        texts.update(json.load(open(DATA_DIR / "Train" / f"{f}-text.json", encoding="utf-8")))
        for k, es in json.load(open(DATA_DIR / "Train" / f"{f}-gold.json", encoding="utf-8")).items():
            gold[k] = [[e[0], e[1], e[2]] for e in es]
    test = json.load(open(DATA_DIR / "Test" / "fold3-text.json", encoding="utf-8"))
    split = json.load(open(SPLIT_FILE, encoding="utf-8"))
    return texts, gold, test, split


# ----------------------------------------------------------------------------- tokens + labels
class Doc:
    """One tokenized document: ids, character spans per token, and (if gold given) BIO tag ids."""
    def __init__(self, doc_id, text, tok, spans_gold=None, split_punct=True):
        self.id, self.text = doc_id, text
        ids, spans = encode_with_offsets(tok, text, split_punct)
        self.ids = np.asarray(ids, dtype=np.int64)
        self.spans = np.asarray(spans, dtype=np.int64).reshape(-1, 2)
        self.labels = None if spans_gold is None else self._bio(spans_gold)

    def _bio(self, ents):
        """Token t belongs to entity e if its (whitespace-stripped) span overlaps e.
        Pure-whitespace tokens inside an entity (e.g. double spaces) get I- so the entity is not cut."""
        lab = np.zeros(len(self.ids), dtype=np.int64)
        starts, ends = self.spans[:, 0], self.spans[:, 1]
        for s, e, l in ents:
            lo = np.searchsorted(ends, s, side="right")          # first token ending after s
            hi = np.searchsorted(starts, e, side="left")         # tokens starting before e
            first = True
            for t in range(lo, hi):
                ts, te = starts[t], ends[t]
                if te > ts and (te <= s or ts >= e):
                    continue
                if te == ts and not (s < ts < e):                # whitespace token outside the entity
                    continue
                if te == ts and first:
                    continue
                lab[t] = b_id(l) if first else i_id(l)
                first = False
        return lab


def make_windows(n_tokens, max_len, stride):
    """Start offsets of windows of length <= max_len covering [0, n_tokens) with the given stride."""
    if n_tokens <= max_len:
        return [0]
    starts = list(range(0, n_tokens - max_len, stride)) + [n_tokens - max_len]
    return sorted(set(starts))


def window_plan(n, L, stride_frac=0.5, shift=0.0):
    """Windows (start, length) of length <= L covering a document of n tokens.
    shift in [0,1): random offset of the windows after the first (training augmentation)."""
    if n <= L:
        return [(0, n)]
    stride = max(1, int(L * stride_frac))
    off = int(shift * stride)
    return [(min(s + off, n - L) if s else s, L) for s in make_windows(n, L, stride)]


def centrality(start, length, n_tokens):
    """How far each position of a window is from a window edge that is NOT a document edge.
    Used to keep, for every token, the prediction of the window where it has the most context."""
    i = np.arange(length)
    left = i if start > 0 else np.full(length, 10**9)
    right = (length - 1 - i) if start + length < n_tokens else np.full(length, 10**9)
    return np.minimum(left, right)


# ----------------------------------------------------------------------------- decoding
def _allowed_matrix():
    """Transition constraints for BIO: I-X may follow only B-X or I-X."""
    A = np.zeros((N_TAGS, N_TAGS), dtype=bool)               # A[prev, cur]
    for p in range(N_TAGS):
        for c in range(N_TAGS):
            tc = TAGS[c]
            if tc.startswith("I-"):
                A[p, c] = TAGS[p] in ("B-" + tc[2:], "I-" + tc[2:])
            else:
                A[p, c] = True
    start_ok = np.array([not t.startswith("I-") for t in TAGS])
    return A, start_ok


_ALLOWED, _START_OK = _allowed_matrix()


def viterbi_bio(logp, skip=None):
    """Most likely tag sequence that is valid BIO. logp: [T, N_TAGS] log-probabilities.
    skip: bool [T] tokens that are ignored (pure whitespace): they inherit the previous tag decision."""
    T = logp.shape[0]
    if T == 0:
        return np.zeros(0, dtype=np.int64)
    keep = np.arange(T) if skip is None else np.where(~skip)[0]
    if len(keep) == 0:
        return np.zeros(T, dtype=np.int64)
    lp = logp[keep]
    trans = np.where(_ALLOWED, 0.0, -1e9)
    score = np.where(_START_OK, lp[0], -1e9)
    back = np.zeros((len(keep), N_TAGS), dtype=np.int64)
    for t in range(1, len(keep)):
        cand = score[:, None] + trans                         # [prev, cur]
        back[t] = cand.argmax(0)
        score = cand.max(0) + lp[t]
    path = np.zeros(len(keep), dtype=np.int64)
    path[-1] = score.argmax()
    for t in range(len(keep) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    out = np.zeros(T, dtype=np.int64)
    out[keep] = path
    if skip is not None:                                       # whitespace tokens: I-X if inside an entity
        for t in np.where(skip)[0]:
            nxt = t + 1
            while nxt < T and skip[nxt]:
                nxt += 1
            if nxt < T and TAGS[out[nxt]].startswith("I-"):
                out[t] = out[nxt]
    return out


def tags_to_spans(doc, tags):
    """BIO tag ids -> [[start, end, label], ...] character spans (whitespace trimmed)."""
    ents, cur = [], None
    for t, tg in enumerate(tags):
        name = TAGS[tg]
        s, e = int(doc.spans[t, 0]), int(doc.spans[t, 1])
        if name == "O":
            if cur: ents.append(cur); cur = None
        elif name.startswith("B-") or cur is None or cur[2] != name[2:]:
            if cur: ents.append(cur)
            cur = [s, e, name[2:]] if e > s else None
        else:
            if e > s:
                cur[1] = e
    if cur: ents.append(cur)
    out = []
    for s, e, l in ents:
        while s < e and doc.text[s].isspace(): s += 1
        while e > s and doc.text[e - 1].isspace(): e -= 1
        if e > s:
            out.append([s, e, l])
    return snap_to_words(doc.text, out)


def snap_to_words(text, ents):
    """Extend every span that starts or ends inside a word to the whole word, then merge spans that now
    overlap (label of the longer original piece wins). Gold spans almost never cut a word (36 of 1,628 val
    entities), but the model sometimes tags only some BPE pieces of a word ('Th' + 'ukral' for 'Thukral',
    'OF 200' for 'OF 2008'). +0.005 to +0.017 val macro-F1 on every Phase-2 model."""
    cand = []
    for s, e, l in ents:
        w = e - s
        while s > 0 and text[s - 1].isalnum() and text[s].isalnum():
            s -= 1
        while e < len(text) and text[e].isalnum() and text[e - 1].isalnum():
            e += 1
        cand.append([s, e, l, w])
    cand.sort(key=lambda x: (x[0], -x[1]))
    merged = []
    for s, e, l, w in cand:
        if merged and s < merged[-1][1]:
            m = merged[-1]
            if w > m[3]:
                m[2], m[3] = l, w
            m[1] = max(m[1], e)
        else:
            merged.append([s, e, l, w])
    return [[s, e, l] for s, e, l, _ in merged]
