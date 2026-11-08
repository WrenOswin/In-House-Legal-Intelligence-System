"""Builds WORK/cache once: splits.pkl, meta.json and the memory-mapped token files of every split for both
Phase-1 tokenizers (35k: A_d768 / A_gqa_d576, 32k: A_d560, B, C, D), then prints token-length statistics.
    python build_cache.py"""
import time

import numpy as np

import lsi
from backbone import BACKBONES, PHASE1_RUNS
from bpe import BPETokenizer

TOKENIZERS = {"A_d768": "bpe_v35000_666b72a523.json", "A_d560": "bpe_v32000_f346ed5f65.json"}

if __name__ == "__main__":
    t0 = time.time()
    m = lsi.meta()
    print({s: len(m[s]["ids"]) for s in lsi.SPLITS}, f"{time.time()-t0:.0f}s", flush=True)
    for bb, tn in TOKENIZERS.items():
        tok = BPETokenizer.load(PHASE1_RUNS / BACKBONES[bb] / tn)
        for s in lsi.SPLITS:
            t1 = time.time()
            T = lsi.Tokens(tn[:-5], s, tok)
            L = np.diff(T.off)
            q = np.percentile(L, [10, 50, 90, 99]).astype(int)
            ch = np.ceil(L / 512)
            print(f"{tn} {s}: {len(T)} docs, {L.sum()/1e6:.1f}M tokens, p10/50/90/99 {q.tolist()} max {L.max()} | "
                  f"512-chunks/doc mean {ch.mean():.2f}, <=4 {np.mean(ch<=4):.3f}, <=8 {np.mean(ch<=8):.3f} | "
                  f"{time.time()-t1:.0f}s", flush=True)
