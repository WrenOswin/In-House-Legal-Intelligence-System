"""Sequence-level KD data: the finished teacher run summarizes every train doc -> <teacher run>/train_gen.json
({doc id: summary}). Resumable (saves every 10 batches), throttled like training.
    python teacher_gen.py <teacher run dir> [max_docs]
"""
import json
import sys
import time
from pathlib import Path

import torch

from train import DEFAULTS, Throttle, be_nice, load_pipeline

if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    max_docs = int(sys.argv[2]) if len(sys.argv) > 2 else None
    out_f = run_dir / "train_gen.json"
    done = json.load(open(out_f, encoding="utf-8")) if out_f.exists() else {}
    cfg = {**DEFAULTS, **json.load(open(run_dir / "config.json", encoding="utf-8"))}
    be_nice(cfg)
    throttle = Throttle(cfg["duty"])
    pipe = load_pipeline(run_dir)
    data = pipe.data["train"]
    idx = [i for i in range(len(data)) if data[i]["id"] not in done][:None if max_docs is None else max(0, max_docs - len(done))]
    idx.sort(key=lambda i: -pipe.n_tokens(pipe.source("train", i), []))
    bs = cfg["gen_batch"]
    t0 = time.time()
    print(f"teacher {run_dir.name}: {len(done)} done, {len(idx)} to go", flush=True)
    for b, s in enumerate(range(0, len(idx), bs)):
        ids = idx[s:s + bs]
        for i, text in zip(ids, pipe.generate("train", ids, throttle)):
            done[data[i]["id"]] = text
        if b % 10 == 9 or s + bs >= len(idx):
            json.dump(done, open(out_f, "w", encoding="utf-8"), ensure_ascii=False)
            el = time.time() - t0
            print(f"  {s + len(ids)}/{len(idx)} docs | {el/60:.1f} min | ~{el / (s + len(ids)) * (len(idx) - s - len(ids)) / 60:.0f} min left",
                  flush=True)
    if max_docs is None or len(done) >= max_docs:
        (run_dir / "train_gen.done").write_text(f"{len(done)} docs\n", encoding="utf-8")
    torch.cuda.empty_cache()
