"""Frozen Phase-1 backbones (no training) as retrievers on dev: doc vs maxsim scoring, alone and fused with
BM25. Reference point for what fine-tuning adds.
    python zeroshot.py A_d768 [eval_chunks]"""
import sys
import time

import numpy as np
import torch

import pcr
from backbone import load_backbone
from model import ChunkEncoder, pad_sets, score_blocks

DEVICE = "cuda"


def zscore(s):
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-6)


@torch.no_grad()
def pool_sets(enc, toks, pad, n_chunks, token_budget=8192):
    rows, owner = [], []
    for i in range(len(toks)):
        cs, _ = pcr.chunks(toks[i], 512, n_chunks)
        rows += cs; owner += [i] * len(cs)
    vecs = torch.cat([enc(rows[s:s + 512], pad, DEVICE, token_budget) for s in range(0, len(rows), 512)])
    return pad_sets(vecs, owner, len(toks))


def fuse_report(neural, bm, split="dev"):
    out = {}
    for w in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
        out[w] = pcr.evaluate(zscore(bm) + w * zscore(neural), split)["f1"]
    return out


if __name__ == "__main__":
    name = sys.argv[1]
    n_chunks = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    bb, tok, info = load_backbone(name, bidirectional=False, device=DEVICE)
    bb.eval()
    enc = ChunkEncoder(bb).to(DEVICE).eval()
    from pathlib import Path
    toks = pcr.Tokens(Path(info["tokenizer"]).stem, "dev", tok)
    t0 = time.time()
    sets, mask = pool_sets(enc, toks, tok.pad_id, n_chunks)
    m = pcr.meta()["dev"]
    q = torch.as_tensor(m["query_idx"], device=DEVICE)
    bm = pcr.bm25_scores("dev", 1.2, 1.0, -3)
    print(f"{name}: encoded {len(toks)} dev docs in {time.time()-t0:.0f}s, {int(mask.sum())} chunks")
    for kind in ("doc", "maxsim"):
        s = score_blocks(sets[q], mask[q], sets, mask, kind).float().cpu().numpy()
        print(kind, pcr.evaluate(s), "| fused with BM25 (w: F1):", fuse_report(s, bm))
