"""
Ensemble of finished runs: average their per-sentence label probabilities (CRF marginals saved in probs.pkl)
and take the most probable label per sentence. No training.
Optional logit adjustment (Menon et al. 2021): label = argmax log p - tau * log prior(train label frequency),
which moves predictions from frequent labels (RatioOfTheDecision) to rare ones; macro-F1 rewards that.

    python ensemble.py NAME RUN_DIR [RUN_DIR ...]     -> dev scores + Phase3/submissions/E_ensemble/<stamp>_NAME/
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from evaluate import format_scores, rr_scores, write_submission
from rr_data import DOMAINS, LABELS, PROJECT_DIR, label_counts, load_rr


def average(run_dirs, split):
    probs = [pickle.load(open(Path(r) / "probs.pkl", "rb")).get(split) for r in run_dirs]
    if any(p is None for p in probs):                  # full-data runs have no dev predictions
        return None
    return {k: np.mean([p[k] for p in probs], 0) for k in probs[0]}


def to_labels(avg, log_prior=None, tau=0.0):
    out = {}
    for k, v in avg.items():
        score = np.log(v + 1e-9)
        if tau:
            score = score - tau * log_prior
        out[k] = [LABELS[i] for i in score.argmax(1)]
    return out


def run(name, run_dirs, make_submission=True, tau=0.0):
    data = load_rr()
    prior = label_counts(data["train"]).astype(float) + 1
    lp = np.log(prior / prior.sum())
    dev = data["dev"]
    gold = {d.id: [LABELS[i] for i in d.labels] for d in dev}
    avg = average(run_dirs, "dev")
    if avg is None:
        sc = dict(macro_f1_all=None, macro_f1_present=None)
        print(f"ensemble {name} ({len(run_dirs)} runs): includes full-data runs -> no dev score")
    else:
        sc = rr_scores(to_labels(avg, lp, tau), gold, {d.id: DOMAINS[d.domain] for d in dev})
        print(f"ensemble {name} ({len(run_dirs)} runs)\n" + format_scores(sc))
    if not make_submission:
        return sc, None
    sub_dir = PROJECT_DIR / "submissions" / "E_ensemble" / f"submission_{time.strftime('%Y%m%d_%H%M%S')}_{name}"
    info = dict(name=name, runs=[str(r) for r in run_dirs], dev_macro_f1_all=sc["macro_f1_all"],
                dev_macro_f1_present=sc["macro_f1_present"], tau=tau,
                decoding="argmax of averaged CRF marginals" + (f" - {tau} * log prior" if tau else ""))
    z = write_submission(to_labels(average(run_dirs, "test"), lp, tau), data["test"], sub_dir, info)
    print("submission:", z)
    return sc, z


if __name__ == "__main__":
    # python ensemble.py NAME [--tau 0.5] RUN_DIR ...
    args = sys.argv[2:]
    tau = 0.0
    if args and args[0] == "--tau":
        tau, args = float(args[1]), args[2:]
    run(sys.argv[1], args, tau=tau)
