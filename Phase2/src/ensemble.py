"""
Ensembling: average the per-token tag probabilities of several finished runs that share a tokenizer,
then decode once (BIO Viterbi + word snapping).

    logp = cached_logprobs(run_dir, split)   # {doc_id: [T, N_TAGS]} on 'val' or 'test', cached in the run dir
    pred = ensemble_predict([run_a, run_b], split)
"""
import json
from pathlib import Path

import numpy as np
import torch

from ner_data import Doc, load_split
from train_ner import DEVICE, build_model, decode_doc, predict_logprobs


def _docs(cfg, tok, split):
    texts, gold, test, sp = load_split()
    if split == "val":
        return [Doc(k, texts[k], tok, gold[k], cfg["split_punct"]) for k in sp["val"]]
    return [Doc(k, test[k], tok, None, cfg["split_punct"]) for k in test]


def cached_logprobs(run_dir, split):
    """Per-document log-probabilities of one run on 'val' or 'test' (computed once, saved as npz)."""
    run_dir = Path(run_dir)
    cache = run_dir / f"logprobs_{split}.npz"
    cfg = json.load(open(run_dir / "config.json", encoding="utf-8"))
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}, cfg
    model, tok, info = build_model(cfg)
    model.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=DEVICE))
    out = {d.id: predict_logprobs(model, d, info["max_len"], cfg["infer_stride_frac"]) for d in _docs(cfg, tok, split)}
    np.savez_compressed(cache, **out)
    del model
    torch.cuda.empty_cache()
    return out, cfg


def ensemble_predict(run_dirs, split, weights=None):
    """Average the probabilities (not log-probabilities) of the runs, then decode each document."""
    lps, cfgs = zip(*[cached_logprobs(r, split) for r in run_dirs])
    toks = {c["backbone"] for c in cfgs}
    from backbone import load_backbone
    _, tok, info = load_backbone(cfgs[0]["backbone"])
    for c in cfgs[1:]:
        _, t2, i2 = load_backbone(c["backbone"])
        assert i2["tokenizer"] == info["tokenizer"], "ensemble members must share a tokenizer"
    w = np.asarray(weights or [1.0] * len(run_dirs), dtype=np.float64)
    w = w / w.sum()
    docs = _docs(cfgs[0], tok, split)
    pred = {}
    for d in docs:
        p = sum(wi * np.exp(lp[d.id].astype(np.float64)) for wi, lp in zip(w, lps))
        pred[d.id] = decode_doc(d, np.log(np.clip(p, 1e-12, 1.0)))
    return pred


def ensemble_submission(run_dirs, tag, roles=True, propagate_names=True, val_scores=None):
    """Ensemble (+ post-processing) predictions on the 35 test docs -> Phase2/submissions/E7_ensemble/."""
    import time
    from evaluate import write_submission
    from ner_data import PROJECT_DIR
    from postprocess import NAME_LABELS, propagate_strings, resolve_party_roles
    _, _, test, _ = load_split()
    pred = ensemble_predict(run_dirs, "test")
    if roles:
        pred = {d: resolve_party_roles(test[d], p) for d, p in pred.items()}
    if propagate_names:
        pred = {d: propagate_strings(test[d], p, labels=NAME_LABELS) for d, p in pred.items()}
    out = PROJECT_DIR / "submissions" / "E7_ensemble" / f"submission_{time.strftime('%Y%m%d_%H%M%S')}_{tag}"
    info = dict(members=[str(r) for r in run_dirs], averaging="mean of per-token tag probabilities",
                decoding="BIO Viterbi + word snapping", party_roles=roles, propagate_names=propagate_names,
                val_scores=val_scores)
    z = write_submission({d: [e[:3] for e in p] for d, p in pred.items()}, test, out, info)
    print(f"submission: {z} ({sum(len(v) for v in pred.values())} entities)")
    return z
