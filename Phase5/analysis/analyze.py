"""Phase-5 report material from the saved val logits of every finished run (CPU only, no training):
    python analyze.py            -> analysis/results.md + results.csv
  1. results table (params, cost, every thresholding strategy, chosen strategy)
  2. per-statute F1 vs train frequency (frequency quintiles) for each run
  3. labels per doc / F1 by gold set size (docs with 1, 2-3, 4-6, 7+ statutes)
  4. most missed statutes and most frequent false-positive statutes of the best run
  5. co-occurrence: for the best run, how often a predicted pair partner is right / missed
  6. document length: F1 by length quintile (number of 512-token chunks)
"""
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import lsi
from run_queue import RUNS, run_dir
from train import sigmoid

OUT = Path(__file__).resolve().parent


def macro(pred, gold):
    return lsi.scores(pred, gold)["macro_f1"]


def main():
    yv, ytr = lsi.label_matrix("val"), lsi.label_matrix("train")
    freq = ytr.sum(0)
    tok = lsi.Tokens("bpe_v32000_f346ed5f65", "val")
    nchunk = np.ceil(np.diff(tok.off) / 512).astype(int)
    rows, runs = [], {}
    for name in RUNS:
        rd = run_dir(name)
        if not (rd / "result.json").exists():
            continue
        r = json.load(open(rd / "result.json", encoding="utf-8"))
        lg = pickle.load(open(rd / "logits.pkl", "rb"))
        thr = np.load(rd / "thresholds.npy")
        pv = sigmoid(lg["val"]); pred = lsi.decode(pv, thr)
        runs[name] = (pv, pred, r)
        rep = r["thresholds_report"]; sc = lsi.scores(pred, yv)
        rows.append(dict(run=name, exp=r["exp"], backbone=r["backbone"], method=r["method"], head=r["head"],
                         total_M=round(r["total_params"] / 1e6, 2), trainable_M=round(r["trainable_params"] / 1e6, 3),
                         train_min=r["train_minutes"], peak_vram_gb=r.get("peak_vram_gb"),
                         val_infer_s=r["val_infer_sec"], f1_05=rep["f1_05"], f1_global=rep["f1_global"],
                         f1_label_cv=rep["f1_label_s0.0_cv"], f1_shrunk_cv=rep["f1_label_s0.5_cv"],
                         chosen=r["threshold_strategy"], f1_chosen_fit=sc["macro_f1"], micro_f1=sc["micro_f1"],
                         subset_acc=sc["subset_acc"], labels_per_doc=sc["labels_per_doc"]))
    with open(OUT / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    md = ["# Phase 5 results (full val, 10,037 docs)\n", "## 1. Runs\n",
          "| run | arch | method | head | params M (trainable) | train min | VRAM GB | @0.5 | global | per-label cv | shrunk cv | chosen (fit) | micro | subset |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['run']} | {r['backbone']} | {r['method']} | {r['head']} | {r['total_M']} ({r['trainable_M']}) | "
                  f"{r['train_min']} | {r['peak_vram_gb']} | {r['f1_05']:.4f} | {r['f1_global']:.4f} | {r['f1_label_cv']:.4f} | "
                  f"{r['f1_shrunk_cv']:.4f} | {r['chosen']} ({r['f1_chosen_fit']:.4f}) | {r['micro_f1']:.4f} | {r['subset_acc']:.4f} |")

    # 2. frequency quintiles
    q = np.argsort(np.argsort(freq)) * 5 // len(freq)            # 0 = rarest 20 statutes
    md += ["\n## 2. Per-statute F1 by train frequency (quintiles of 20 statutes, chosen thresholds)\n",
           "| run | " + " | ".join(f"Q{k+1} ({int(freq[q==k].min())}-{int(freq[q==k].max())} docs)" for k in range(5)) + " |",
           "|---|" + "---|" * 5]
    for name, (pv, pred, r) in runs.items():
        f1 = lsi.per_label_f1(pred, yv)[0]
        md.append(f"| {name} | " + " | ".join(f"{f1[q==k].mean():.3f}" for k in range(5)) + " |")
    corr = {n: float(np.corrcoef(np.log(freq), lsi.per_label_f1(p, yv)[0])[0, 1]) for n, (_, p, _) in runs.items()}
    md.append("\nPearson r(log train frequency, per-statute F1): " + ", ".join(f"{n} {c:.2f}" for n, c in corr.items()))

    # 3. gold set size + 6. length
    ng = yv.sum(1)
    bins = [(1, 1), (2, 3), (4, 6), (7, 99)]
    md += ["\n## 3. By number of gold statutes per doc (macro-F1 within the group; pred labels/doc vs gold)\n",
           "| run | " + " | ".join(f"{a}-{b} ({int(((ng>=a)&(ng<=b)).sum())} docs)" for a, b in bins) + " |", "|---|" + "---|" * len(bins)]
    for name, (pv, pred, r) in runs.items():
        cells = []
        for a, b in bins:
            m = (ng >= a) & (ng <= b)
            cells.append(f"{macro(pred[m], yv[m]):.3f} ({pred[m].sum(1).mean():.1f} vs {ng[m].mean():.1f})")
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    lb = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 10**6)]
    md += ["\n## 6. By document length (512-token chunks; inference reads <= 8)\n",
           "| run | " + " | ".join(f"{a}-{b if b < 10**6 else '+'} ({int(((nchunk>=a)&(nchunk<=b)).sum())})" for a, b in lb) + " |",
           "|---|" + "---|" * len(lb)]
    for name, (pv, pred, r) in runs.items():
        md.append(f"| {name} | " + " | ".join(f"{macro(pred[(nchunk>=a)&(nchunk<=b)], yv[(nchunk>=a)&(nchunk<=b)]):.3f}"
                                              for a, b in lb) + " |")

    # 4./5. best run
    best = max(runs, key=lambda n: runs[n][2]["thresholds_report"]["f1_label_s0.5_cv"])
    pv, pred, r = runs[best]
    f1, p, rc = lsi.per_label_f1(pred, yv)
    fn = (~pred & yv.astype(bool)).sum(0); fp = (pred & ~yv.astype(bool)).sum(0)
    md += [f"\n## 4. Best run {best}: weakest and strongest statutes\n", "| statute | train docs | val docs | F1 | P | R | FN | FP |",
           "|---|---|---|---|---|---|---|---|"]
    for l in list(np.argsort(f1)[:12]) + list(np.argsort(f1)[-5:]):
        md.append(f"| {l} | {int(freq[l])} | {int(yv[:, l].sum())} | {f1[l]:.3f} | {p[l]:.3f} | {rc[l]:.3f} | {int(fn[l])} | {int(fp[l])} |")
    # confusions: for each missed gold statute, which wrong statute was predicted most often in the same docs
    G = yv.astype(bool)
    conf = {}
    for l in np.argsort(fn)[::-1][:10]:
        m = G[:, l] & ~pred[:, l]
        wrong = (pred[m] & ~G[m]).sum(0)
        conf[int(l)] = [(int(k), int(wrong[k])) for k in np.argsort(wrong)[::-1][:3] if wrong[k] > 0]
    md += ["\nMost-missed statutes and the wrong statute most often predicted instead (same docs):",
           *[f"- {l}: missed {int(fn[l])}x; instead {v}" for l, v in conf.items()]]
    # co-occurrence
    co = (ytr.T.astype(np.float64) @ ytr) / np.maximum(1, freq)[:, None]      # P(b | a) in train
    np.fill_diagonal(co, 0)
    pairs = [(a, b) for a in range(100) for b in range(100) if co[a, b] >= 0.6 and freq[a] >= 300]
    hit = miss = 0
    for a, b in pairs:
        m = pred[:, a] & G[:, a] & G[:, b]
        hit += (m & pred[:, b]).sum(); miss += (m & ~pred[:, b]).sum()
    md += [f"\n## 5. Co-occurrence ({len(pairs)} statute pairs with P(b|a) >= 0.6 in train)\n",
           f"When the best run correctly predicts a and b is also gold: b predicted {hit}x, missed {miss}x "
           f"({hit / max(1, hit + miss):.1%} recall on partners vs overall micro recall "
           f"{(pred & G).sum() / G.sum():.1%})."]
    (OUT / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
