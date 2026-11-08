"""Tokenize train/val/test sentence by sentence with both Phase-1 tokenizers (32k: A d560/B/C/D, 35k: A d768).
    .venv python build_cache.py
"""
import pickle
import time
from pathlib import Path

import summ
from bpe import BPETokenizer

RUNS = Path("C:/kaggle/working/cs781/runs")
TOKENIZERS = [RUNS / "A_decoder_only/d768_L12_T1024_v35k_5fbdf6/bpe_v35000_666b72a523.json",
              RUNS / "A_decoder_only/d560_L8_T1024_v32k_6613f2/bpe_v32000_f346ed5f65.json"]

if __name__ == "__main__":
    sp = summ.build_splits()
    summ.salience_scores()
    for tf in TOKENIZERS:
        tok = BPETokenizer.load(tf)
        for split in ("val", "test", "train"):
            p = summ.CACHE / f"tok_{tf.stem}_{split}.pkl"
            if p.exists():
                continue
            t = time.time()
            data = summ.tokenize_split(tok, sp[split])
            pickle.dump(data, open(p, "wb"))
            n_doc = sum(sum(len(s) for s in d["sents"]) for d in data)
            n_sum = sum(len(d["summary"]) for d in data if d["summary"] is not None)
            words = sum(len(d["document"].split()) for d in sp[split])
            print(f"{tf.stem} {split}: {len(data)} docs, {n_doc} doc tokens ({n_doc / words:.3f}/word), "
                  f"{n_sum} summary tokens, {time.time() - t:.0f}s", flush=True)
