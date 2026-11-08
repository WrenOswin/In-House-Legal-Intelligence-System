"""Phase-6 comparison of all finished runs (val generations saved by train.py) -> analysis/results.md, results.csv
    .venv python analysis/analyze.py
Per run: ROUGE-1/2/L, length, cost (params, trainable, VRAM, time) and, from the 300 val generations:
  coverage     share of the reference's content-word types present in the generated summary
  copy rate    share of generated word bigrams that occur in the source document (1 - abstractiveness)
  novel bi.    share of generated bigrams not in the document (reference summaries: 0.33)
  num. halluc. share of numbers (>= 2 digits) in the summary that never occur in the document (factuality proxy)
  rep4 / rep-sent  repeated 4-grams / sentences inside one summary (redundancy)
  R1 by length   ROUGE-1 for docs whose reference is short / medium / long (terciles)
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import summ  # noqa: E402

RUNS = summ.WORK / "runs"
OUT = Path(__file__).parent
NUM = re.compile(r"\d[\d,./]*\d")


def bigrams(t):
    w = t.lower().split()
    return list(zip(w, w[1:]))


def stats(preds, docs, refs):
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)
    cov, copy, hall, rsent, r1 = [], [], [], [], []
    for p, d, r in zip(preds, docs, refs):
        rw = set(summ.content_words(r))
        cov.append(len(rw & set(summ.content_words(p))) / max(1, len(rw)))
        db = set(bigrams(d)); pb = bigrams(p)
        copy.append(sum(b in db for b in pb) / max(1, len(pb)))
        nums = [n for n in NUM.findall(p) if len(re.sub(r"\D", "", n)) >= 2]
        dn = set(re.sub(r"\D", "", n) for n in NUM.findall(d))
        hall.append(sum(re.sub(r"\D", "", n) not in dn for n in nums) / len(nums) if nums else np.nan)
        ss = [s.strip().lower() for s in re.split(r"(?<=[.?!])\s+", p) if len(s.split()) >= 5]
        rsent.append(1 - len(set(ss)) / len(ss) if ss else 0.0)
        r1.append(sc.score(r, p)["rouge1"].fmeasure)
    rl = np.array([len(r.split()) for r in refs])
    t1, t2 = np.percentile(rl, [33.3, 66.7])
    r1 = np.array(r1)
    return dict(coverage=np.mean(cov), copy_rate=np.mean(copy), novel_bigrams=1 - np.mean(copy),
                num_halluc=float(np.nanmean(hall)), rep_sent=np.mean(rsent),
                r1_short=r1[rl <= t1].mean(), r1_mid=r1[(rl > t1) & (rl <= t2)].mean(), r1_long=r1[rl > t2].mean())


def main():
    sp = summ.build_splits()
    val = {d["id"]: d for d in sp["val"]}
    rows = []
    for rf in sorted(RUNS.glob("S*/*/result.json")):
        res = json.load(open(rf, encoding="utf-8"))
        gen = json.load(open(rf.parent / "val_gen.json", encoding="utf-8"))
        ids = list(gen)
        st = stats([gen[i] for i in ids], [val[i]["document"] for i in ids], [val[i]["summary"] for i in ids])
        v = res["val_scores"]
        rows.append(dict(exp=res["exp"], run_id=res["run_id"], method=res["method_desc"].split(":")[0].split(" (")[0],
                         params_M=res["total_params"] / 1e6, trainable_M=res["trainable_params"] / 1e6,
                         rouge1=v["rouge1"], rouge2=v["rouge2"], rougeL=v["rougeL"], words=v["words"], rep4=v["rep4"],
                         **st, train_min=res["train_minutes"], tok_s=res["tok_per_s"], vram_gb=res["peak_vram_gb"],
                         val_gen_s=res["val_gen_sec"], best_val_loss=res["best_val_loss"]))
    # the reference summaries themselves, for scale
    refs = [d["summary"] for d in sp["val"]]
    ref_st = stats(refs, [d["document"] for d in sp["val"]], refs)
    with open(OUT / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    L = ["# Phase 6 results (val, 300 held-out train docs, default decoding)", "",
         "## Table 1 - quality and cost", "",
         "| run | method | params M | trainable M | ROUGE-1 | ROUGE-2 | ROUGE-L | words | train min | tok/s | VRAM GB | val gen s |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['exp']} | {r['method']} | {r['params_M']:.1f} | {r['trainable_M']:.2f} | {r['rouge1']:.4f} | "
                 f"{r['rouge2']:.4f} | {r['rougeL']:.4f} | {r['words']:.0f} | {r['train_min']:.1f} | {r['tok_s']} | "
                 f"{r['vram_gb']:.2f} | {r['val_gen_s']:.0f} |")
    L += ["", "## Table 2 - coverage, abstractiveness, factuality proxy, redundancy", "",
          "| run | coverage | novel bigrams | num. halluc. | rep4 | repeated sent. | R1 short ref | R1 mid | R1 long |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows + [dict(exp="reference summaries", rep4=summ.repetition(refs, 4), **ref_st)]:
        L.append(f"| {r['exp']} | {r['coverage']:.3f} | {r['novel_bigrams']:.3f} | {r['num_halluc']:.3f} | {r['rep4']:.4f} | "
                 f"{r['rep_sent']:.3f} | {r['r1_short']:.3f} | {r['r1_mid']:.3f} | {r['r1_long']:.3f} |")
    L += ["", "coverage = share of the reference's content words found in the output; num. halluc. = share of numbers "
          "(>= 2 digits) in the output that never occur in the document; R1 by reference-length tercile.", ""]
    (OUT / "results.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
