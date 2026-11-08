"""Finish a run that was interrupted after at least one eval (power cut, killed process): loads its
model_best.pt, scores the full val, picks thresholds, predicts test and writes the submission + result.json,
exactly like the end of train.run_experiment (training-time stats come from history.json).
    python finalize.py K5            (queue name)"""
import json
import pickle
import sys
import time

import numpy as np

import lsi
from run_queue import resolve, run_dir
from train import DEFAULTS, Throttle, be_nice, choose_thresholds, load_pipeline, sigmoid

if __name__ == "__main__":
    name = sys.argv[1]
    cfg = {**DEFAULTS, **resolve(name)}
    rd = run_dir(name)
    assert (rd / "model_best.pt").exists() and not (rd / "result.json").exists(), rd
    be_nice(cfg)
    throttle = Throttle(cfg["duty"])
    hist = json.load(open(rd / "history.json", encoding="utf-8"))
    best = max(hist, key=lambda h: h["f1_global"])
    pipe = load_pipeline(rd)
    m, yva = lsi.meta(), lsi.label_matrix("val")
    val_idx = list(range(len(yva)))
    t1 = time.time(); val_lg = pipe.predict("val", val_idx, throttle); val_sec = time.time() - t1
    strat, thr, rep = choose_thresholds(sigmoid(val_lg), yva)
    sct = lsi.scores(lsi.decode(sigmoid(val_lg), thr), yva)
    t1 = time.time(); te_lg = pipe.predict("test", list(range(len(m["test"]["ids"]))), throttle)
    pred = lsi.decode(sigmoid(te_lg), thr)
    from peft import count_params
    total, trainable = count_params(pipe.model)
    res = dict(exp=cfg["exp"], run_id=rd.name, backbone=cfg["backbone"], method=cfg["method"], head=cfg["head"],
               loss=cfg["loss"], method_desc=pipe.info["method_desc"], total_params=total, trainable_params=trainable,
               best_at=(best["epoch"], best["step"]), train_minutes=hist[-1]["minutes"], interrupted=True,
               val_infer_sec=round(val_sec, 1), test_infer_sec=round(time.time() - t1, 1), val_docs=len(val_idx),
               thresholds_report=rep, threshold_strategy=strat, val_scores_fit=sct, teacher=cfg.get("kd_teacher"),
               test_labels_per_doc=round(float(pred.sum(1).mean()), 2))
    sub = lsi.PROJECT_DIR / "submissions" / cfg["exp"] / f"submission_{time.strftime('%Y%m%d_%H%M%S')}_{rd.name}"
    res["submission"] = str(lsi.write_submission(lsi.pred_dict(pred), sub, dict(res, config=cfg, thresholds=thr.tolist())))
    np.save(rd / "thresholds.npy", thr)
    pickle.dump({"val": val_lg, "val_idx": val_idx, "test": te_lg}, open(rd / "logits.pkl", "wb"))
    json.dump(res, open(rd / "result.json", "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps(rep), "\nchosen", strat, sct, "\n", res["submission"])
