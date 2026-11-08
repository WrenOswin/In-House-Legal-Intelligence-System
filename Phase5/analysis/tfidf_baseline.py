"""Reference only (not a submission, no Phase-1 model): TF-IDF (1-2 grams) + one-vs-rest logistic regression on
the Phase-5 splits, scored on val with the same thresholding strategies as the neural runs.
    python tfidf_baseline.py      (CPU, ~15-30 min)"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lsi

t0 = time.time()
sp = lsi.build_splits()
ytr, yva = lsi.label_matrix("train"), lsi.label_matrix("val")
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True, dtype=np.float32)
Xtr = vec.fit_transform(d["text"] for d in sp["train"]); Xva = vec.transform(d["text"] for d in sp["val"])
print(f"tfidf {Xtr.shape} {time.time()-t0:.0f}s", flush=True)
clf = OneVsRestClassifier(LogisticRegression(C=4.0, solver="liblinear", class_weight="balanced"), n_jobs=1)   # n_jobs>1: joblib read-only memmap error
clf.fit(Xtr, ytr)
pv = clf.predict_proba(Xva)
print(f"fit {time.time()-t0:.0f}s", flush=True)
rep = lsi.threshold_report(pv, yva)
print(rep)
tg = rep["t_global"]
print("global:", lsi.scores(lsi.decode(pv, tg), yva))
print("per-label (fit):", lsi.scores(lsi.decode(pv, lsi.label_thresholds(pv, yva, 0.0, tg)), yva))
np.save(lsi.WORK / "tfidf_val_prob.npy", pv.astype(np.float32))
