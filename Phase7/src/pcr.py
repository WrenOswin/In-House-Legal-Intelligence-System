"""
Phase-7 Prior Case Retrieval data, BM25, metrics and submission writing (Phase-5 lsi.py layout):
  * every split = one candidate pool (train / dev / test) + the queries of that split; every query is ALSO a
    document of its own pool (checked: same id) -> a query is never allowed to retrieve itself
  * documents are stored once per pool (the queries are pool documents), so each split is tokenized once
  * tokens come from memory-mapped .npy files (built once per tokenizer and split)

Normalisation (same for every split): sentences joined with spaces, whitespace collapsed, case kept, and the
dataset's mask strings mapped to words that are a single Phase-1 BPE token (the masks are 3-4 tokens each):
  <NAME> -> X   <ORG> -> Y   <CITATION> -> citation

Metric (Codabench): micro-F1@7 over all queries, each query gets exactly 7 unique candidates of its pool.
"""
import json
import pickle
import re
import zipfile
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]                 # .../Capstone Project/Phase7
DATA_DIR = Path("C:/kaggle/working/cs781/hf_data/Phase_7_PCR")
WORK = Path("C:/kaggle/working/cs781/phase7")
SPLITS = ("train", "dev", "test")
FILES = {"train": ("Train/train.json", "Train/train_candidates.json"),
         "dev": ("Dev/dev.json", "Dev/dev_candidates.json"),
         "test": ("Test/test.json", "Test/test_candidates.json")}
K = 7
_WS = re.compile(r"\s+")
MASKS = {"<NAME>": "X", "<ORG>": "Y", "<CITATION>": "citation"}
_MASK_RE = re.compile("|".join(re.escape(k) for k in MASKS))


def normalise(sents):
    text = " ".join(sents) if isinstance(sents, list) else sents
    return _WS.sub(" ", _MASK_RE.sub(lambda m: MASKS[m.group(0)], text)).strip()


def build_splits():
    """Raw JSON -> {split: dict(doc_ids, texts, query_ids, gold)} (WORK/cache/splits.pkl; built once).
    gold[i] = list of pool indices relevant to query i (empty for test)."""
    f = WORK / "cache" / "splits.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    out = {}
    for s, (qf, cf) in FILES.items():
        q = json.load(open(DATA_DIR / qf, encoding="utf-8"))
        c = json.load(open(DATA_DIR / cf, encoding="utf-8"))
        ids = list(c)
        pos = {d: i for i, d in enumerate(ids)}
        assert all(qi in pos for qi in q), f"{s}: query outside its pool"
        texts = [normalise(c[d]["text"]) for d in ids]
        qids = list(q)
        gold = [sorted(pos[g] for g in q[qi].get("relevant_candidates", [])) for qi in qids]
        assert all(pos[qi] not in g for qi, g in zip(qids, gold)), f"{s}: a query is its own gold"
        out[s] = dict(doc_ids=ids, texts=texts, query_ids=qids, query_idx=[pos[qi] for qi in qids], gold=gold)
    f.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(f, "wb"))
    return out


def meta():
    """Everything except the texts - small, cached in WORK/cache/meta.json."""
    f = WORK / "cache" / "meta.json"
    if not f.exists():
        sp = build_splits()
        json.dump({s: {k: v for k, v in sp[s].items() if k != "texts"} for s in SPLITS}, open(f, "w", encoding="utf-8"))
    return json.load(open(f, encoding="utf-8"))


class Tokens:
    """Memory-mapped token ids of one pool for one tokenizer: tokens[i] -> int32 array of document i."""
    def __init__(self, tok_name, split, tok=None):
        base = WORK / "cache" / f"tokens_{tok_name}_{split}"
        flat_f, off_f = base.with_suffix(".flat.npy"), base.with_suffix(".off.npy")
        if not flat_f.exists():                     # tokenize once (needs the tokenizer)
            texts = build_splits()[split]["texts"]
            parts = [np.array(tok.encode_ordinary(t), dtype=np.int32) for t in texts]
            off = np.zeros(len(parts) + 1, dtype=np.int64); off[1:] = np.cumsum([len(p) for p in parts])
            np.save(flat_f, np.concatenate(parts)); np.save(off_f, off)
        self.flat = np.load(flat_f, mmap_mode="r")
        self.off = np.load(off_f)
        assert len(self) == len(meta()[split]["doc_ids"]), f"{base} out of date"

    def __len__(self):
        return len(self.off) - 1

    def __getitem__(self, i):
        return np.asarray(self.flat[self.off[i]:self.off[i + 1]])


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


# ---------------------------------------------------------------- ranking + metrics
def rank(scores, query_idx, k=K):
    """scores [n_queries, n_pool] -> top-k pool indices per query (the query itself excluded)."""
    s = np.array(scores, dtype=np.float64, copy=True)
    s[np.arange(len(s)), query_idx] = -np.inf
    top = np.argpartition(-s, k, axis=1)[:, :k]
    order = np.take_along_axis(s, top, 1).argsort(1)[:, ::-1]
    return np.take_along_axis(top, order, 1)


def f1_at_k(top, gold, k=K):
    """Micro P/R/F1@k (Codabench): tp summed over queries / (k * n_queries) and / sum |gold|."""
    tp = sum(len(set(t[:k].tolist()) & set(g)) for t, g in zip(top, gold))
    n_pred, n_gold = k * len(top), sum(len(g) for g in gold)
    p, r = tp / n_pred, tp / max(1, n_gold)
    return dict(f1=round(2 * p * r / (p + r) if tp else 0.0, 4), p=round(p, 4), r=round(r, 4))


def evaluate(scores, split="dev"):
    m = meta()[split]
    top = rank(scores, m["query_idx"])
    out = f1_at_k(top, m["gold"])
    for k in (1, 3, 5, 10, 20):                  # extra cut-offs for the report (recall@k, precision@k)
        t = rank(scores, m["query_idx"], k)
        out[f"r@{k}"] = f1_at_k(t, m["gold"], k)["r"]
    return out


# ---------------------------------------------------------------- BM25 (from scratch, scipy sparse)
_WORD = re.compile(r"[a-z][a-z0-9]+")
STOP = set("""the of and to in a is that for be by on as it this was with not or are which an at from has have
he his been were any under such its no they there their had said would may also we if all so but than other
shall these into can upon only could should them who what when where being our one two did does""".split())


def words(text):
    return [w for w in _WORD.findall(text.lower()) if w not in STOP]


def grams(ws, ngram):
    """ngram=n: word n-grams 1..n; ngram=-n: only n-grams of length n."""
    ns = [-ngram] if ngram < 0 else range(1, ngram + 1)
    return [" ".join(ws[i:i + n]) for n in ns for i in range(len(ws) - n + 1)]


def bm25_scores(split, k1=1.2, b=0.75, ngram=1):
    """[n_queries, n_pool] BM25 scores of every query of a split against its own pool (pool = BM25 corpus).
    ngram: see grams(). Cached in WORK/cache/bm25_<split>_<k1>_<b>_<ngram>.npy."""
    f = WORK / "cache" / f"bm25_{split}_{k1}_{b}_{ngram}.npy"
    if f.exists():
        return np.load(f)
    from scipy import sparse
    sp = build_splits()[split]
    vocab, rows, cols, vals = {}, [], [], []
    for i, t in enumerate(sp["texts"]):
        ws = grams(words(t), ngram)
        cnt = {}
        for w in ws:
            j = vocab.setdefault(w, len(vocab)); cnt[j] = cnt.get(j, 0) + 1
        rows += [i] * len(cnt); cols += list(cnt); vals += list(cnt.values())
    tf = sparse.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(len(sp["texts"]), len(vocab)))
    dl = np.asarray(tf.sum(1)).ravel(); avg = dl.mean()
    df = np.bincount(tf.indices, minlength=len(vocab))
    idf = np.log(1 + (len(dl) - df + 0.5) / (df + 0.5)).astype(np.float32)
    w = tf.tocoo()
    sat = w.data * (k1 + 1) / (w.data + k1 * (1 - b + b * dl[w.row] / avg))
    D = sparse.csr_matrix((sat * idf[w.col], (w.row, w.col)), shape=tf.shape)          # doc term weights
    Q = (tf[sp["query_idx"]] > 0).astype(np.float32)                                      # query terms (binary)
    s = (Q @ D.T).toarray().astype(np.float32)
    np.save(f, s)
    return s


# ---------------------------------------------------------------- submission
def write_submission(top, sub_dir, info=None, split="test"):
    """prediction.json = {query_id: [7 candidate ids, best first]} for every test query, zipped at the root."""
    m = meta()[split]
    ids = m["doc_ids"]
    pred = {qi: [ids[j] for j in row[:K]] for qi, row in zip(m["query_ids"], top)}
    pool = set(ids)
    assert len(pred) == len(m["query_ids"]) == (237 if split == "test" else len(pred))
    assert all(len(v) == K == len(set(v)) and set(v) <= pool and q not in v for q, v in pred.items())
    sub_dir = Path(sub_dir); sub_dir.mkdir(parents=True, exist_ok=True)
    pj = sub_dir / "prediction.json"
    json.dump(pred, open(pj, "w", encoding="utf-8"), indent=1)
    z = sub_dir / "submission.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pj, "prediction.json")
    if info is not None:
        json.dump(info, open(sub_dir / "run_info.json", "w", encoding="utf-8"), indent=1, default=str)
    return z
