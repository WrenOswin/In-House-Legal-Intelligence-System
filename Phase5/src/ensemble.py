"""
Ensembles of finished Phase-5 runs - CPU only. Averages each run's sigmoid probabilities on val and test
(logits.pkl), picks the thresholding strategy by cross-fitted val macro-F1, writes a submission.

    python ensemble.py K1 K5            (runs by queue name)
    python ensemble.py --all-subsets    (val score of every subset of finished runs, no files written)
"""
import itertools
import pickle
import sys
import time

import numpy as np

import lsi
from run_queue import RUNS, run_dir
from train import choose_thresholds, sigmoid


def probs(name):
    lg = pickle.load(open(run_dir(name) / "logits.pkl", "rb"))
    assert len(lg["val"]) == len(lsi.meta()["val"]["ids"]), f"{name}: val logits are not on the full val split"
    return sigmoid(lg["val"]), sigmoid(lg["test"])


def evaluate(names):
    yv = lsi.label_matrix("val")
    pv, pt = zip(*(probs(n) for n in names))
    pv, pt = np.mean(pv, 0), np.mean(pt, 0)
    strat, thr, rep = choose_thresholds(pv, yv)
    return lsi.scores(lsi.decode(pv, thr), yv), strat, thr, rep, pt


if __name__ == "__main__":
    finished = [n for n in RUNS if (run_dir(n) / "logits.pkl").exists()]
    if sys.argv[1:] == ["--all-subsets"]:
        for k in range(1, len(finished) + 1):
            for sub in itertools.combinations(finished, k):
                sc, strat, _, rep, _ = evaluate(sub)
                print(f"{'+'.join(sub):20s} global {rep['f1_global']:.4f} | per-label cv {rep['f1_label_s0.0_cv']:.4f}"
                      f" | shrunk cv {rep['f1_label_s0.5_cv']:.4f} | chosen {strat}")
        sys.exit()
    names = sys.argv[1:]
    sc, strat, thr, rep, pt = evaluate(names)
    pred = lsi.decode(pt, thr)
    tag = "+".join(names)
    sub = lsi.PROJECT_DIR / "submissions" / "E_ensemble" / f"submission_{time.strftime('%Y%m%d_%H%M%S')}_{tag}"
    z = lsi.write_submission(lsi.pred_dict(pred), sub, dict(members=[str(run_dir(n)) for n in names],
                                                             strategy=strat, thresholds=thr.tolist(), report=rep, val=sc))
    print(f"{tag}: {rep}\nchosen {strat}: {sc} | test labels/doc {pred.sum(1).mean():.2f}\n{z}")
