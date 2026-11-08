"""
Phase-6 data, long-document context selection, metrics and submission files.

Data: Phase_6_Summ/Train/train.json (7,030 {id, document, summary}) and Test/test.json (100 {id, document}).
No dev split is given -> a fixed held-out val split of VAL_SIZE train docs (seeded; documents that also occur
elsewhere in train are never put in val, so val text is never seen in training).

Documents are split into sentences; every sentence is tokenized separately with the Phase-1 BPE (tokens cached
per tokenizer), so a context of any token budget can be assembled from whole sentences:
  head      first sentences until the budget is full
  headtail  first 2/3 of the budget from the start, last 1/3 from the end (judgment's operative part)
  select    importance-based selection: a ridge model (hashed word/bigram TF-IDF + position + length features)
            trained on train docs predicts how much of each sentence reappears in the reference summary;
            the top sentences are kept in document order (the first sentence is always kept - case header)
  chunks    the whole document cut into consecutive token chunks (B's hierarchical / Fusion-in-Decoder input)
"""
import json
import pickle
import random
import re
import zipfile
from pathlib import Path

import numpy as np

HF = Path("C:/kaggle/working/cs781/hf_data/Phase_6_Summ")
WORK = Path("C:/kaggle/working/cs781/phase6")
CACHE = WORK / "cache"
PROJECT = Path(__file__).resolve().parents[1]
VAL_SIZE, SPLIT_SEED = 300, 6

_WS = re.compile(r"\s+")
# sentence boundary: . ? ! ; or : followed by space and an upper-case letter / digit / bracket,
# not after common legal abbreviations (No. Nos. S. Ss. Art. Vol. Mr. etc.) or single initials
_ABBR = r"(?<!\bNo)(?<!\bNos)(?<!\bS)(?<!\bSs)(?<!\bArt)(?<!\bArts)(?<!\bVol)(?<!\bMr)(?<!\bMrs)(?<!\bDr)" \
        r"(?<!\bJ)(?<!\bJJ)(?<!\bCo)(?<!\bLtd)(?<!\bvs)(?<!\bv)(?<!\bp)(?<!\bpp)(?<!\bpara)(?<!\bsec)(?<!\bcl)" \
        r"(?<!\b[A-Z])"
_SENT = re.compile(_ABBR + r"(?<=[.?!])\s+(?=[A-Z0-9(\"'])")
MAX_SENT_CHARS = 1200                                       # very long "sentences" are cut into pieces


def normalise(text):
    return _WS.sub(" ", text).strip()


def sentences(text):
    out = []
    for s in _SENT.split(normalise(text)):
        while len(s) > MAX_SENT_CHARS:                      # cut at the last space before the limit
            cut = s.rfind(" ", 0, MAX_SENT_CHARS)
            cut = cut if cut > 200 else MAX_SENT_CHARS
            out.append(s[:cut]); s = s[cut:].lstrip()
        if s:
            out.append(s)
    return out


def load_raw():
    tr = json.load(open(HF / "Train/train.json", encoding="utf-8"))
    te = json.load(open(HF / "Test/test.json", encoding="utf-8"))
    return tr, te


def build_splits():
    """{'train': [...], 'val': [...], 'test': [...]} of dicts id/document/summary (normalised)."""
    p = CACHE / "splits.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    tr, te = load_raw()
    for d in tr + te:
        d["document"] = normalise(d["document"])
        if "summary" in d:
            d["summary"] = normalise(d["summary"])
    count = {}
    for d in tr:
        count[d["document"]] = count.get(d["document"], 0) + 1
    unique = [i for i, d in enumerate(tr) if count[d["document"]] == 1 and len(d["summary"].split()) >= 30]
    rng = random.Random(SPLIT_SEED)
    val_idx = set(rng.sample(unique, VAL_SIZE))
    splits = dict(train=[d for i, d in enumerate(tr) if i not in val_idx],
                  val=[d for i, d in enumerate(tr) if i in val_idx],
                  test=[dict(id=d["id"], document=d["document"]) for d in te])
    CACHE.mkdir(parents=True, exist_ok=True)
    pickle.dump(splits, open(p, "wb"))
    return splits


# ------------------------------------------------------------------ tokens (cached per tokenizer)
def tokenize_split(tok, docs):
    """-> list of dicts: sents (list of int32 arrays, one per sentence, each sentence after the first encoded
    with a leading space), summary (int32 array or None)."""
    out = []
    for d in docs:
        ss = sentences(d["document"])
        sents = [np.asarray(tok.encode_ordinary(s if k == 0 else " " + s), dtype=np.int32) for k, s in enumerate(ss)]
        summ = np.asarray(tok.encode_ordinary(d["summary"]), dtype=np.int32) if d.get("summary") else None
        out.append(dict(id=d["id"], sents=sents, text_sents=ss, summary=summ))
    return out


def load_tokens(tok_file, split):
    """tok_file: file name of the Phase-1 tokenizer (bpe_v32000_... / bpe_v35000_...)."""
    return pickle.load(open(CACHE / f"tok_{Path(tok_file).stem}_{split}.pkl", "rb"))


# ------------------------------------------------------------------ context selection
def _take(sents, order, budget):
    keep, used = [], 0
    for i in order:
        n = len(sents[i])
        if used + n > budget:
            if used == 0:                                    # a single giant sentence: cut it
                keep.append((i, n - (used + n - budget))); used = budget
            continue
        keep.append((i, n)); used += n
        if used >= budget:
            break
    keep.sort()
    if not keep:
        return np.zeros(0, dtype=np.int32)
    return np.concatenate([sents[i][:n] for i, n in keep])


def context(doc, budget, strategy="select", scores=None):
    """Token array (<= budget) built from whole sentences of a tokenized doc."""
    sents = doc["sents"]
    n = len(sents)
    if strategy == "head":
        return _take(sents, range(n), budget)
    if strategy == "headtail":
        head = _take(sents, range(n), budget * 2 // 3)
        used = 0; k = 0
        for k in range(n):
            used += len(sents[k])
            if used > len(head):
                break
        tail_ids, t_used = [], 0
        for i in range(n - 1, k - 1, -1):
            if t_used + len(sents[i]) > budget - len(head):
                break
            tail_ids.append(i); t_used += len(sents[i])
        tail = np.concatenate([sents[i] for i in sorted(tail_ids)]) if tail_ids else np.zeros(0, np.int32)
        return np.concatenate([head, tail])
    if strategy == "select":
        assert scores is not None and len(scores) == n
        order = [0] + [int(i) for i in np.argsort(-np.asarray(scores)) if i != 0]
        return _take(sents, order, budget)
    raise ValueError(strategy)


def chunks(doc, chunk_len, max_chunks, scores=None):
    """Whole document as consecutive chunks of <= chunk_len tokens (sentence-aligned). If there are more
    than max_chunks, keep the max_chunks chunks with the highest mean sentence score (or evenly spaced)."""
    out, cur, cur_sc = [], [], []
    for i, s in enumerate(doc["sents"]):
        s = s[:chunk_len]
        if sum(len(x) for x in cur) + len(s) > chunk_len and cur:
            out.append((np.concatenate(cur), float(np.mean(cur_sc)))); cur, cur_sc = [], []
        cur.append(s); cur_sc.append(scores[i] if scores is not None else 0.0)
    if cur:
        out.append((np.concatenate(cur), float(np.mean(cur_sc))))
    if len(out) > max_chunks:
        if scores is not None:
            keep = sorted([0] + [int(i) for i in np.argsort([-c[1] for c in out]) if i != 0][:max_chunks - 1])
        else:
            keep = sorted(set(np.linspace(0, len(out) - 1, max_chunks).round().astype(int).tolist()))
        out = [out[i] for i in keep]
    return [c for c, _ in out]


# ------------------------------------------------------------------ importance model (sentence salience)
_WORD = re.compile(r"[a-z]+")
STOP = set("""a an the of to in and or for on by with as at from that this these those is are was were be been being
it its which who whom whose there their they he she his her him we our you your i not no nor but if then than so such
any all each other into upon under over said also may shall would should could can will has have had do does did""".split())


def content_words(text):
    return [w for w in _WORD.findall(text.lower()) if w not in STOP and len(w) > 2]


def salience_target(sent, summary_words):
    """Share of the sentence's content words (types) that occur in the reference summary."""
    ws = set(content_words(sent))
    return len(ws & summary_words) / len(ws) if ws else 0.0


def _pos_features(n):
    rel = (np.arange(n) + 0.5) / max(n, 1)
    bins = np.minimum((rel * 20).astype(int), 19)
    f = np.zeros((n, 24), dtype=np.float32)
    f[np.arange(n), bins] = 1.0
    f[:, 20] = rel
    f[:min(n, 3), 21] = 1.0                                  # first sentences (header)
    f[max(0, n - 3):, 22] = 1.0                              # last sentences (order)
    f[:, 23] = np.log1p(n) / 8
    return f


class Salience:
    """Ridge regression on hashed TF-IDF (word uni+bigrams) + position/length features -> salience_target."""
    def __init__(self):
        from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
        self.hv = HashingVectorizer(n_features=2 ** 18, ngram_range=(1, 2), alternate_sign=False,
                                    norm=None, lowercase=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
        self.tfidf = TfidfTransformer(sublinear_tf=True)
        self.model = None

    def _features(self, sents, ns, fit=False):
        import scipy.sparse as sp
        X = self.hv.transform(sents)
        X = self.tfidf.fit_transform(X) if fit else self.tfidf.transform(X)
        lens = np.array([len(s.split()) for s in sents], dtype=np.float32)
        extra = np.concatenate([_pos_features_multi(ns), (np.log1p(lens) / 5)[:, None]], 1)
        return sp.hstack([X, sp.csr_matrix(extra)]).tocsr()

    def fit(self, docs, max_docs=4000, seed=0):
        from sklearn.linear_model import Ridge
        rng = random.Random(seed)
        docs = rng.sample(docs, min(max_docs, len(docs)))
        sents, y, ns = [], [], []
        for d in docs:
            ss = sentences(d["document"])
            sw = set(content_words(d["summary"]))
            sents += ss; ns.append(len(ss)); y += [salience_target(s, sw) for s in ss]
        X = self._features(sents, ns, fit=True)
        self.model = Ridge(alpha=3.0, solver="sag", max_iter=200, random_state=0).fit(X, np.asarray(y))
        return self

    def score(self, text_sents):
        return self.model.predict(self._features(text_sents, [len(text_sents)]))


def _pos_features_multi(ns):
    return np.concatenate([_pos_features(n) for n in ns], 0)


def salience_scores():
    """{split: [np.array of sentence scores per doc]} - fitted once on train, cached."""
    p = CACHE / "salience.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    sp = build_splits()
    model = Salience().fit(sp["train"])
    out = {k: [model.score(sentences(d["document"])) for d in v] for k, v in sp.items()}
    pickle.dump(out, open(p, "wb"))
    return out


# ------------------------------------------------------------------ metrics
def rouge(preds, refs):
    """Mean ROUGE-1/2/L F1 (rouge_score, no stemming - the HF `evaluate` default)."""
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    acc = {"rouge1": [], "rouge2": [], "rougeL": []}
    for p, r in zip(preds, refs):
        s = sc.score(r, p)
        for k in acc:
            acc[k].append(s[k].fmeasure)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def repetition(texts, n=4):
    """Share of repeated word n-grams per text (0 = no repetition), averaged."""
    out = []
    for t in texts:
        w = t.split()
        g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
        out.append(1 - len(set(g)) / len(g) if g else 0.0)
    return float(np.mean(out))


def coverage(contexts, refs):
    """Share of the reference summary's content-word types that appear in the given input text."""
    out = []
    for c, r in zip(contexts, refs):
        rw = set(content_words(r))
        out.append(len(rw & set(content_words(c))) / len(rw) if rw else 0.0)
    return float(np.mean(out))


# ------------------------------------------------------------------ submission
def write_submission(pred_by_id, sub_dir, info=None):
    """prediction.json in the list format [{"id", "summary"}] + submission.zip (prediction.json at root)."""
    sp = build_splits()
    ids = [d["id"] for d in sp["test"]]
    assert set(pred_by_id) == set(ids), "need exactly one summary per test doc"
    rows = [{"id": i, "summary": pred_by_id[i].strip() or "The appeal was disposed of."} for i in ids]
    sub_dir = Path(sub_dir)
    sub_dir.mkdir(parents=True, exist_ok=True)
    with open(sub_dir / "prediction.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    with zipfile.ZipFile(sub_dir / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(sub_dir / "prediction.json", "prediction.json")
    if info is not None:
        json.dump(info, open(sub_dir / "info.json", "w", encoding="utf-8"), indent=1)
    return sub_dir
