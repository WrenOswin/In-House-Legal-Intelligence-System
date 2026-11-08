"""Decoding study on a finished run (no training): generation length and repetition controls on val docs.
    python decode_sweep.py <run dir> [n_val_docs]                     grid -> <run>/decode_sweep.json
    python decode_sweep.py <run dir> --test '{"max_new": 384, ...}'    test summaries with these settings -> new submission
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

import summ
from train import DEFAULTS, GEN_KEYS, Throttle, be_nice, load_pipeline, score_texts

GRID = [  # (max_new, min_new, rep_penalty, no_repeat)
    (256, 128, 1.0, 4), (384, 200, 1.0, 4), (512, 200, 1.0, 4), (512, 400, 1.0, 4),
    (512, 200, 1.0, 0), (512, 200, 1.0, 6), (512, 200, 1.1, 4), (512, 200, 1.2, 4),
]

if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    cfg = {**DEFAULTS, **json.load(open(run_dir / "config.json", encoding="utf-8"))}
    be_nice(cfg)
    throttle = Throttle(cfg["duty"])
    pipe = load_pipeline(run_dir)
    sp = summ.build_splits()
    if len(sys.argv) > 3 and sys.argv[2] == "--test":
        gen = json.loads(sys.argv[3])
        res = json.load(open(run_dir / "result.json", encoding="utf-8"))
        idx = list(range(len(pipe.data["test"])))
        preds = pipe.generate("test", idx, throttle, **gen)
        by_id = {pipe.data["test"][i]["id"]: p for i, p in zip(idx, preds)}
        tag = "_".join(f"{k}{gen[k]}" for k in GEN_KEYS if k in gen)
        sub = summ.PROJECT / "submissions" / cfg["exp"] / f"submission_{time.strftime('%Y%m%d_%H%M%S')}_{res['run_id']}_{tag}"
        info = dict(res, decoding={**res["decoding"], **gen}, test_words=float(np.mean([len(p.split()) for p in preds])),
                    config=cfg)
        print(summ.write_submission(by_id, sub, info), f"{info['test_words']:.0f} words/summary")
        sys.exit()
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    idx = list(range(n))
    refs = [sp["val"][i]["summary"] for i in idx]
    out_f = run_dir / "decode_sweep.json"
    rows = json.load(open(out_f, encoding="utf-8")) if out_f.exists() else []
    have = {(r["max_new"], r["min_new"], r["rep_penalty"], r["no_repeat"], r["n"]) for r in rows}
    for mx, mn, rp, nr in GRID:
        if (mx, mn, rp, nr, n) in have:
            continue
        t = time.time()
        preds = pipe.generate("val", idx, throttle, max_new=mx, min_new=mn, rep_penalty=rp, no_repeat=nr)
        sc = score_texts(preds, refs)
        rows.append(dict(max_new=mx, min_new=mn, rep_penalty=rp, no_repeat=nr, n=n, sec=round(time.time() - t), **sc))
        print(json.dumps(rows[-1]), flush=True)
        json.dump(rows, open(out_f, "w", encoding="utf-8"), indent=1)
