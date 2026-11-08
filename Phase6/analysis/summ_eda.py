"""Phase-6 data analysis + extractive reference baselines (no Phase-1 model). Output: analysis/data_eda.txt
    .venv python analysis/summ_eda.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import summ  # noqa: E402

OUT = Path(__file__).with_name("data_eda.txt")
lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); lines.append(s)


def pct(a):
    a = np.asarray(a)
    return " ".join(f"{q}%={np.percentile(a, q):.0f}" for q in (10, 25, 50, 75, 90, 99)) + f" max={a.max():.0f} mean={a.mean():.0f}"


sp = summ.build_splits()
say("splits", {k: len(v) for k, v in sp.items()})
for k in ("train", "val", "test"):
    say(f"{k} doc words  ", pct([len(d["document"].split()) for d in sp[k]]))
    if k != "test":
        say(f"{k} summ words ", pct([len(d["summary"].split()) for d in sp[k]]))
        say(f"{k} ratio %    ", pct([100 * len(d["summary"].split()) / len(d["document"].split()) for d in sp[k]]))
say("sentences/doc (val)", pct([len(summ.sentences(d["document"])) for d in sp["val"]]))

# novelty: share of summary word bigrams that are not in the document (abstractiveness)
def bigrams(t):
    w = t.lower().split()
    return set(zip(w, w[1:]))
nov = [1 - len(bigrams(d["summary"]) & bigrams(d["document"])) / max(1, len(bigrams(d["summary"]))) for d in sp["val"]]
say("novel summary bigrams (val) mean %.3f median %.3f" % (np.mean(nov), np.median(nov)))

# where does summary content come from? salience target by relative position decile
dec = np.zeros(10); cnt = np.zeros(10)
for d in sp["val"]:
    ss = summ.sentences(d["document"]); sw = set(summ.content_words(d["summary"]))
    for i, s in enumerate(ss):
        b = min(9, int(10 * i / len(ss))); dec[b] += summ.salience_target(s, sw); cnt[b] += 1
say("mean sentence salience by position decile:", " ".join(f"{x:.3f}" for x in dec / cnt))

sal = summ.salience_scores()
refs = [d["summary"] for d in sp["val"]]

# word-level contexts approximating token budgets (1 BPE token ~ 0.8 words, measured below)
def words_ctx(d, scores, budget_words, strategy):
    ss = summ.sentences(d["document"])
    fake = dict(sents=[np.zeros(len(s.split()), np.int32) for s in ss])
    # reuse summ.context on word counts, then map kept sentences back
    keep_mask = []
    if strategy == "select":
        order = [0] + [int(i) for i in np.argsort(-scores) if i != 0]
    else:
        order = range(len(ss))
    used, kept = 0, []
    for i in order:
        n = len(ss[i].split())
        if used + n > budget_words:
            continue
        kept.append(i); used += n
    return " ".join(ss[i] for i in sorted(kept))

say("\ncoverage of reference content words by the model input (val):")
say("  whole document          %.3f" % summ.coverage([d["document"] for d in sp["val"]], refs))
for bw in (400, 800, 1200, 1600):
    for st in ("head", "select"):
        ctx = [words_ctx(d, sal["val"][i], bw, st) for i, d in enumerate(sp["val"])]
        say(f"  {st:6s} {bw:5d} words      %.3f" % summ.coverage(ctx, refs))

say("\nextractive baselines on val (ROUGE F1, no Phase-1 model):")
for bw in (300, 500, 700):
    lead = [" ".join(d["document"].split()[:bw]) for d in sp["val"]]
    sel = [words_ctx(d, sal["val"][i], bw, "select") for i, d in enumerate(sp["val"])]
    say(f"  lead-{bw} words   ", summ.rouge(lead, refs))
    say(f"  select-{bw} words ", summ.rouge(sel, refs))
OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
