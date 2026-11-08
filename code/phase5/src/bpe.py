"""Byte-level BPE tokenizer - copied verbatim from the Phase-1 notebooks (cell "2. Tokenizer"),
plus Phase-2 helpers at the bottom: encoding with character offsets."""
import os, json, time, heapq
from collections import Counter, defaultdict
from pathlib import Path
import multiprocessing as mp
import regex as re_u

HAVE_REGEX = True
HAVE_FORK = "fork" in mp.get_all_start_methods()

SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|bos|>", "<|sep|>", "<|mask|>"]

def make_pattern(split_digits=True):
    num = r"\p{N}{1,3}" if split_digits else r"\p{N}+"
    if HAVE_REGEX:
        return (r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?""" + num +
                r"""| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")
    # fallback (no \p classes in stdlib re)
    d = r"\d{1,3}" if split_digits else r"\d+"
    return r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?""" + d + r"""| ?[^\s\w]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

def _merge_ids(ids, pair, new_id):
    out, i, n = [], 0, len(ids)
    a, b = pair
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out

_COUNT_PAT = None
def _count_words(texts_and_pattern):
    texts, pattern = texts_and_pattern
    global _COUNT_PAT
    if _COUNT_PAT is None or _COUNT_PAT.pattern != pattern:
        _COUNT_PAT = re_u.compile(pattern)
    c = Counter()
    for t in texts:
        c.update(_COUNT_PAT.findall(t))
    return c


class BPETokenizer:
    def __init__(self, merges=None, pattern=None, specials=SPECIAL_TOKENS):
        self.pattern = pattern or make_pattern()
        self.pat = re_u.compile(self.pattern)
        self.merges = [tuple(m) for m in (merges or [])]
        self.specials = list(specials)
        self._build()

    # ---------- vocab bookkeeping ----------
    def _build(self):
        self.ranks = {p: i for i, p in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        base = 256 + len(self.merges)
        self.special_to_id = {s: base + i for i, s in enumerate(self.specials)}
        self.id_to_special = {v: k for k, v in self.special_to_id.items()}
        self.vocab_size = base + len(self.specials)
        self._special_re = (re_u.compile("(" + "|".join(re_u.escape(s) for s in self.specials) + ")")
                            if self.specials else None)
        self.cache = {}

    def __len__(self):
        return self.vocab_size

    @property
    def eot_id(self): return self.special_to_id["<|endoftext|>"]
    @property
    def pad_id(self): return self.special_to_id["<|pad|>"]
    @property
    def bos_id(self): return self.special_to_id["<|bos|>"]
    @property
    def sep_id(self): return self.special_to_id["<|sep|>"]
    @property
    def mask_id(self): return self.special_to_id["<|mask|>"]

    # ---------- training ----------
    @classmethod
    def train(cls, text_iter, vocab_size, pattern, min_word_freq=2, n_proc=1, verbose=True):
        t0 = time.time()
        # 1) parallel word-type counting
        def batches(it, size=200):
            buf = []
            for t in it:
                buf.append(t)
                if len(buf) == size:
                    yield (buf, pattern); buf = []
            if buf:
                yield (buf, pattern)
        counts = Counter()
        if n_proc > 1 and HAVE_FORK:
            with mp.get_context("fork").Pool(n_proc) as pool:
                for c in pool.imap_unordered(_count_words, batches(text_iter), chunksize=1):
                    counts.update(c)
        else:
            for b in batches(text_iter):
                counts.update(_count_words(b))
        words, freqs = [], []
        for w, c in counts.items():
            if c >= min_word_freq:
                words.append(list(w.encode("utf-8"))); freqs.append(c)
        del counts
        if verbose:
            print(f"[bpe] {len(words):,} word types after min_freq={min_word_freq} ({time.time()-t0:.0f}s)")

        # 2) initial pair statistics + index
        pair_counts = defaultdict(int)
        where = defaultdict(list)
        for i, w in enumerate(words):
            f = freqs[i]
            seen = set()
            for p in zip(w, w[1:]):
                pair_counts[p] += f
                if p not in seen:
                    where[p].append(i); seen.add(p)
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        # 3) greedy merges
        n_merges = vocab_size - 256 - len(SPECIAL_TOKENS)
        assert n_merges > 0, "vocab_size too small"
        merges = []
        while len(merges) < n_merges and heap:
            negc, p = heapq.heappop(heap)
            cur = pair_counts.get(p, 0)
            if cur <= 0 or -negc != cur:          # stale heap entry
                continue
            new_id = 256 + len(merges)
            merges.append(p)
            touched = set()
            for i in set(where.pop(p, ())):
                w = words[i]
                nw = _merge_ids(w, p, new_id)
                if len(nw) == len(w):
                    continue
                f = freqs[i]
                old_pc = Counter(zip(w, w[1:]))
                new_pc = Counter(zip(nw, nw[1:]))
                for q, n in old_pc.items():
                    pair_counts[q] -= n * f; touched.add(q)
                for q, n in new_pc.items():
                    pair_counts[q] += n * f; touched.add(q)
                    if q not in old_pc:
                        where[q].append(i)
                words[i] = nw
            for q in touched:
                c = pair_counts.get(q, 0)
                if c > 0:
                    heapq.heappush(heap, (-c, q))
                else:
                    pair_counts.pop(q, None)
            if verbose and len(merges) % 2000 == 0:
                print(f"[bpe] {len(merges):,}/{n_merges:,} merges ({time.time()-t0:.0f}s)")
        if verbose:
            print(f"[bpe] done: {len(merges):,} merges in {time.time()-t0:.0f}s")
        if len(merges) < n_merges:
            print(f"[bpe] warning: only {len(merges):,}/{n_merges:,} merges possible -> vocab is smaller than requested "
                  "(raise tok_train_chars or lower tok_min_word_freq)")
        return cls(merges=merges, pattern=pattern)

    # ---------- encode / decode ----------
    def _encode_word(self, s):
        hit = self.cache.get(s)
        if hit is not None:
            return hit
        ids = list(s.encode("utf-8"))
        ranks = self.ranks
        while len(ids) >= 2:
            best, best_r = None, None
            for p in zip(ids, ids[1:]):
                r = ranks.get(p)
                if r is not None and (best_r is None or r < best_r):
                    best, best_r = p, r
            if best is None:
                break
            ids = _merge_ids(ids, best, 256 + best_r)
        if len(self.cache) > 500_000:
            self.cache.clear()
        self.cache[s] = ids
        return ids

    def encode_ordinary(self, text):
        out = []
        for w in self.pat.findall(text):
            out.extend(self._encode_word(w))
        return out

    def encode(self, text, allow_special=True):
        if not allow_special or self._special_re is None:
            return self.encode_ordinary(text)
        out = []
        for part in self._special_re.split(text):
            if part in self.special_to_id:
                out.append(self.special_to_id[part])
            elif part:
                out.extend(self.encode_ordinary(part))
        return out

    def decode(self, ids, skip_special=True):
        buf, out = [], []
        for i in ids:
            i = int(i)
            if i in self.id_to_special:
                if buf:
                    out.append(b"".join(buf).decode("utf-8", errors="replace")); buf = []
                if not skip_special:
                    out.append(self.id_to_special[i])
            else:
                buf.append(self.vocab[i])
        if buf:
            out.append(b"".join(buf).decode("utf-8", errors="replace"))
        return "".join(out)

    # ---------- persistence ----------
    def save(self, path, meta=None):
        path = Path(path)
        tmp = path.with_suffix(".tmp")
        json.dump({"type": "byte_bpe", "pattern": self.pattern, "specials": self.specials,
                   "merges": self.merges, "vocab_size": self.vocab_size, "meta": meta or {}},
                  open(tmp, "w", encoding="utf-8"))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        d = json.load(open(path, encoding="utf-8"))
        return cls(merges=d["merges"], pattern=d["pattern"], specials=d["specials"])


# =============================================================================
# PHASE 2: encoding with character offsets (for span-level NER)
# =============================================================================
_PUNCT_RUN = re_u.compile(r"[^\s\p{L}\p{N}]")


def _pieces(word, split_punct):
    """Split one pre-tokenizer match into pieces that are BPE-encoded separately.
    split_punct: a run of punctuation like ' ].' becomes ' ]', '.' so an entity can end between them
    (the Phase-1 pattern glues punctuation runs into one pre-token)."""
    if not split_punct or len(word.strip()) <= 1 or not _PUNCT_RUN.match(word.lstrip(" ")):
        return [(0, word)]
    out, i = [], 0
    lead = len(word) - len(word.lstrip(" "))
    body_start = lead
    for j in range(body_start, len(word)):
        ch = word[j]
        if ch in "\r\n":                       # trailing newlines stay together
            out.append((i, word[i:])); return out
        out.append((i, word[i:j + 1])); i = j + 1
    return out


def encode_with_offsets(tok, text, split_punct=True):
    """
    Returns (ids, spans): spans[t] = (start, end) character span of token t in `text`,
    with leading whitespace removed (a token ' Haryana' covers only 'Haryana').
    Pure-whitespace tokens get an empty span (start == end) at their end position.
    decode(ids) == text always holds.
    """
    ids, spans = [], []
    for m in tok.pat.finditer(text):
        w, ws = m.group(), m.start()
        for off, piece in _pieces(w, split_punct):
            cs = ws + off
            b2c = []                                       # byte index -> char index within piece
            for ci, ch in enumerate(piece):
                b2c += [ci] * len(ch.encode("utf-8"))
            b2c.append(len(piece))
            bp = 0
            for i in tok._encode_word(piece):
                n = len(tok.vocab[i])
                s, e = b2c[bp], b2c[min(bp + n, len(b2c) - 1)]
                if bp + n < len(b2c) - 1 and b2c[bp + n] == b2c[bp + n - 1]:
                    e = b2c[bp + n - 1] + 1               # token ends inside a multi-byte char -> include it
                while s < e and piece[s].isspace():
                    s += 1
                ids.append(i); spans.append((cs + s, cs + e)); bp += n
    return ids, spans
