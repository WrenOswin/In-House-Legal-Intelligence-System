"""
Ensembles of finished Phase-4 runs - CPU only, seconds. Averages each run's P(label = 1) on val and test
(logits.pkl), tunes the decision threshold on val, writes a submission.

    python ensemble.py J1 J4            (runs by queue name; latest finished run of each)
    python ensemble.py --all-subsets    (val score of every subset of >= 2 finished runs, no files written)
"""
import itertools
import json
import pickle
import sys
import time

import numpy as np

import cjpe
from run_queue import RUNS
from train import DEFAULTS, run_id_of


def run_dir(name):
    cfg = {**DEFAULTS, **RUNS[name]}
    if cfg.get("kd_teacher"):
        cfg["kd_teacher"] = str(cjpe.WORK / "runs" / "J1_A_lorma" / run_id_of({**DEFAULTS, **RUNS["J1"]}))
    return cjpe.WORK / "runs" / cfg["exp"] / run_id_of(cfg)


def probs(name):
    lg = pickle.load(open(run_dir(name) / "logits.pkl", "rb"))
    sm = lambda x: np.exp(x - x.max(1, keepdims=True)) / np.exp(x - x.max(1, keepdims=True)).sum(1, keepdims=True)
    return sm(lg["val"])[:, 1], sm(lg["test"])[:, 1]


def evaluate(names):
    yv = np.array(cjpe.meta()["val"]["labels"])
    pv, pt = zip(*(probs(n) for n in names))
    pv, pt = np.mean(pv, 0), np.mean(pt, 0)
    thr = cjpe.best_threshold(pv, yv)
    return cjpe.scores(pv >= thr, yv), thr, pt


if __name__ == "__main__":
    finished = [n for n in RUNS if (run_dir(n) / "logits.pkl").exists()]
    if sys.argv[1:] == ["--all-subsets"]:
        for k in range(1, len(finished) + 1):
            for sub in itertools.combinations(finished, k):
                sc, thr, _ = evaluate(sub)
                print(f"{'+'.join(sub):20s} val macro-F1 {sc['macro_f1']:.4f} @thr {thr:.2f}")
        sys.exit()
    names = sys.argv[1:]
    sc, thr, pt = evaluate(names)
    ids = cjpe.meta()["test"]["ids"]
    pred = {i: int(p >= thr) for i, p in zip(ids, pt)}
    tag = "+".join(names)
    sub = cjpe.PROJECT_DIR / "submissions" / "E_ensemble" / f"submission_{time.strftime('%Y%m%d_%H%M%S')}_{tag}"
    z = cjpe.write_submission(pred, sub, dict(members=[str(run_dir(n)) for n in names], threshold=thr, val=sc))
    print(f"{tag}: val {sc} @thr {thr:.2f} | test pred pos {np.mean(list(pred.values())):.3f}\n{z}")
