"""
Sentence-level scoring (macro-F1 / P / R, per label, per domain, confusion matrix), prediction.json
writing with validation, and zipping.

Macro averaging is reported two ways until we know which one Codabench uses:
  macro_f1_all      mean over all 13 labels (a label absent from gold and pred counts as F1 = 0)
  macro_f1_present  mean over labels that occur in gold or pred (sklearn's default with labels=None)
"""
import json
import zipfile
from pathlib import Path

import numpy as np

from rr_data import DOMAINS, LABELS, L2I, N_LABELS


def confusion(pred, gold):
    """pred/gold: {doc_id: [label, ...]} -> confusion matrix [gold, pred] over LABELS."""
    cm = np.zeros((N_LABELS, N_LABELS), dtype=np.int64)
    for d, g in gold.items():
        p = pred[d]
        assert len(p) == len(g), f"{d}: {len(p)} predictions for {len(g)} sentences"
        np.add.at(cm, ([L2I[x] for x in g], [L2I[x] for x in p]), 1)
    return cm


def scores_from_confusion(cm):
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    P = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
    R = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=tp + fn > 0)
    F = np.divide(2 * P * R, P + R, out=np.zeros_like(tp), where=P + R > 0)
    present = (cm.sum(0) + cm.sum(1)) > 0
    return dict(macro_f1_all=F.mean(), macro_f1_present=F[present].mean(),
                macro_p_present=P[present].mean(), macro_r_present=R[present].mean(),
                accuracy=tp.sum() / max(1, cm.sum()),
                per_label={l: dict(p=P[i], r=R[i], f1=F[i], support=int(cm[i].sum()), n_pred=int(cm[:, i].sum()))
                           for i, l in enumerate(LABELS)})


def rr_scores(pred, gold, domain_of=None):
    """Overall scores + the same per domain (domain_of: {doc_id: 'CL' | 'IT'})."""
    cm = confusion(pred, gold)
    sc = scores_from_confusion(cm)
    sc["confusion"] = cm.tolist()
    if domain_of:
        for dom in DOMAINS:
            g = {d: v for d, v in gold.items() if domain_of[d] == dom}
            if g:
                s = scores_from_confusion(confusion(pred, g))
                sc[dom] = {k: s[k] for k in ("macro_f1_all", "macro_f1_present", "accuracy")}
    return sc


def format_scores(sc):
    lines = [f"macro-F1 present {sc['macro_f1_present']:.4f} | all-13 {sc['macro_f1_all']:.4f} | "
             f"macro-P {sc['macro_p_present']:.4f} | macro-R {sc['macro_r_present']:.4f} | acc {sc['accuracy']:.4f}"]
    for dom in DOMAINS:
        if dom in sc:
            lines.append(f"  {dom}: macro-F1 present {sc[dom]['macro_f1_present']:.4f} | all-13 "
                         f"{sc[dom]['macro_f1_all']:.4f} | acc {sc[dom]['accuracy']:.4f}")
    for l, v in sc["per_label"].items():
        lines.append(f"  {l:24s} P {v['p']:.3f} R {v['r']:.3f} F1 {v['f1']:.3f}  (gold {v['support']}, pred {v['n_pred']})")
    return "\n".join(lines)


def validate_prediction(pred, test_docs):
    """Every test doc present, exactly one valid label per sentence, nothing extra."""
    want = {d.id: len(d) for d in test_docs}
    assert set(pred) == set(want), f"doc ids differ: missing {set(want) - set(pred)}, extra {set(pred) - set(want)}"
    for k, n in want.items():
        assert len(pred[k]) == n, f"{k}: {len(pred[k])} labels for {n} sentences"
        bad = {x for x in pred[k] if x not in L2I}
        assert not bad, f"{k}: invalid labels {bad}"


def write_submission(pred, test_docs, sub_dir, info=None):
    """prediction.json + submission.zip (prediction.json at the archive root) + run_info.json."""
    validate_prediction(pred, test_docs)
    sub_dir = Path(sub_dir)
    sub_dir.mkdir(parents=True, exist_ok=True)
    pj = sub_dir / "prediction.json"
    json.dump({d.id: list(pred[d.id]) for d in test_docs}, open(pj, "w", encoding="utf-8"), ensure_ascii=False)
    z = sub_dir / "submission.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pj, "prediction.json")
    if info is not None:
        json.dump(info, open(sub_dir / "run_info.json", "w", encoding="utf-8"), indent=1, default=str)
    return z
