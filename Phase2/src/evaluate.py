"""
Strict span-level scoring (same rule as Codabench: start, end and label must all match exactly),
prediction.json writing/validation and error analysis helpers.
"""
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from ner_data import LABELS


def strict_scores(pred, gold, labels=LABELS):
    """pred/gold: {doc_id: [[s, e, label], ...]}. Per-label P/R/F1 over all docs, macro and micro averages."""
    tp, fp, fn = Counter(), Counter(), Counter()
    for d in gold:
        g = {(int(s), int(e), l) for s, e, l, *_ in gold[d]}
        p = {(int(s), int(e), l) for s, e, l, *_ in pred.get(d, [])}
        for x in p & g: tp[x[2]] += 1
        for x in p - g: fp[x[2]] += 1
        for x in g - p: fn[x[2]] += 1
    per = {}
    for l in labels:
        P = tp[l] / (tp[l] + fp[l]) if tp[l] + fp[l] else 0.0
        R = tp[l] / (tp[l] + fn[l]) if tp[l] + fn[l] else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        per[l] = dict(p=P, r=R, f1=F, tp=tp[l], fp=fp[l], fn=fn[l])
    T, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    mP, mR = T / max(1, T + FP), T / max(1, T + FN)
    return dict(macro_f1=sum(v["f1"] for v in per.values()) / len(labels),
                macro_p=sum(v["p"] for v in per.values()) / len(labels),
                macro_r=sum(v["r"] for v in per.values()) / len(labels),
                micro_f1=2 * mP * mR / max(1e-9, mP + mR), per_label=per)


def error_breakdown(pred, gold):
    """Classify every wrong prediction: right span/wrong label, overlapping boundary error, or spurious;
    and every missed gold entity that has no overlapping prediction at all."""
    c = Counter()
    confusion = Counter()
    for d in gold:
        g = {(s, e): l for s, e, l, *_ in gold[d]}
        p = {(s, e): l for s, e, l, *_ in pred.get(d, [])}
        for (s, e), l in p.items():
            if g.get((s, e)) == l:
                continue
            if (s, e) in g:
                c["label_error"] += 1; confusion[(g[(s, e)], l)] += 1
            elif any(s < ge and gs < e for gs, ge in g):
                c["boundary_error"] += 1
            else:
                c["spurious"] += 1
        for (s, e), l in g.items():
            if not any(s < pe and ps < e for ps, pe in p):
                c["missed"] += 1
    return dict(counts=dict(c), top_confusions=[(f"{a}->{b}", n) for (a, b), n in confusion.most_common(10)])


def format_scores(sc):
    lines = [f"strict macro-F1 {sc['macro_f1']:.4f} | macro-P {sc['macro_p']:.4f} | macro-R {sc['macro_r']:.4f}"
             f" | micro-F1 {sc['micro_f1']:.4f}"]
    for l, v in sc["per_label"].items():
        lines.append(f"  {l:10s} P {v['p']:.3f} R {v['r']:.3f} F1 {v['f1']:.3f}  (tp {v['tp']}, fp {v['fp']}, fn {v['fn']})")
    return "\n".join(lines)


def validate_prediction(pred, test_texts):
    """Checks the evaluator performs: exactly the test ids, [int, int, label] items, valid offsets/labels."""
    assert set(pred) == set(test_texts), f"id mismatch: missing {set(test_texts) - set(pred)}, extra {set(pred) - set(test_texts)}"
    for d, ents in pred.items():
        n = len(test_texts[d])
        for x in ents:
            assert isinstance(x, list) and len(x) in (3, 4), (d, x)
            s, e, l = x[:3]
            assert isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= n, (d, x, n)
            assert l in LABELS, (d, x)
    return True


def write_submission(pred, test_texts, out_dir, info):
    """prediction.json (+ entity text as the optional 4th field) and submission.zip with it at the root."""
    validate_prediction(pred, test_texts)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    full = {d: [[s, e, l, test_texts[d][s:e]] for s, e, l in sorted(ents)] for d, ents in pred.items()}
    json.dump(full, open(out_dir / "prediction.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(info, open(out_dir / "run_info.json", "w", encoding="utf-8"), indent=1, default=str)
    with zipfile.ZipFile(out_dir / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_dir / "prediction.json", arcname="prediction.json")
    with zipfile.ZipFile(out_dir / "submission.zip") as z:
        assert z.namelist() == ["prediction.json"]
    return out_dir / "submission.zip"
