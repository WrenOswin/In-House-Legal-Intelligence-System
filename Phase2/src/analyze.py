"""
Report material for Phase 2: figures (Phase2/analysis/figures/*.png) and an error analysis
(Phase2/analysis/error_analysis/error_analysis.md) built from the kept runs' validation predictions
and the Codabench scores recorded in Phase 2.txt.

    python analyze.py
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from ner_data import LABELS, PROJECT_DIR, load_split, snap_to_words
from evaluate import strict_scores

RUNS = Path("C:/kaggle/working/cs781/phase2/runs")
FIG = PROJECT_DIR / "analysis" / "figures"
ERR = PROJECT_DIR / "analysis" / "error_analysis"

# experiments with a validation score, in report order; Codabench scores as reported (Phase 2.txt)
EXPS = [
    ("E2 LoRMA (A 140M)", "E2_A_lorma/A_d768_lorma_684310", 0.5090),
    ("E1 full FT, bidir (A 140M)", "E1_A_full/A_d768_full_8fcd22", 0.4807),
    ("E1 full FT, causal (A 140M)", "E1_A_full/A_d768_full_5c4900", 0.4790),
    ("E6 KD student (A 58M)", "E6_KD/A_d560_full_kd_35adb4", 0.4684),
    ("E6 student alone (A 58M)", "E6_KD/A_d560_full_4a48da", None),
    ("E5 partial FT (D 54M)", "E5_D_partial/D_d560_partial_c656f6", 0.4628),
    ("E4 adapters (C 58M)", "E4_C_adapter/C_d560_adapter_16dd38", 0.4019),
    ("E3 LoRA (B 42M)", "E3_B_lora/B_d512_lora_e60be9", 0.3631),
]
BEST = "E2_A_lorma/A_d768_lorma_684310"

# reference palette (dataviz skill, light mode) - validated: slots 1-4 pass adjacent CVD/normal-vision checks
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
         "#1c5cab", "#184f95", "#104281", "#0d366b"]

plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2, "text.color": INK,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def style(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load_run(rel):
    d = RUNS / rel
    return (json.load(open(d / "result.json", encoding="utf-8")),
            json.load(open(d / "val_metrics.json", encoding="utf-8")),
            json.load(open(d / "val_predictions.json", encoding="utf-8")))


# ----------------------------------------------------------------------------- figures
def fig_scores(runs):
    rows = [(n, r["val_macro_f1"], t) for (n, rel, t), (r, _, _) in zip(EXPS, runs) if t is not None]
    rows.insert(0, ("E2 LoRMA + word snapping", 0.5185, 0.5174))   # same model, snapped decoding (section 10)
    rows = rows[::-1]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(y, [t for _, _, t in rows], height=0.55, color=S1, label="Codabench test (35 docs)", zorder=2)
    ax.scatter([v for _, v, _ in rows], y, s=46, color=S2, edgecolor=SURFACE, linewidth=1.5, zorder=3,
               label="local validation (14 docs)")
    for yi, (_, v, t) in zip(y, rows):
        ax.text(max(t, v) + 0.006, yi, f"{t:.3f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, [n for n, _, _ in rows])
    ax.set_xlim(0, 0.6)
    ax.set_xlabel("strict macro-F1")
    ax.set_title("Phase 2 L-NER: test score per experiment", loc="left", fontsize=12, color=INK)
    style(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_scores_test_vs_val.png", dpi=160)
    plt.close(fig)


def fig_per_label(runs):
    names = [n for n, _, _ in EXPS]
    M = np.array([[m["scores"]["per_label"][l]["f1"] for l in LABELS] for (_, m, _) in runs])
    order = np.argsort(-M.mean(0))                         # easiest label first
    M = M[:, order]; labs = [LABELS[i] for i in order]
    cmap = LinearSegmentedColormap.from_list("blue", BLUES)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if M[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(labs)), labs, rotation=0, fontsize=8.5)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    ax.set_xticks(np.arange(-.5, len(labs)), minor=True); ax.set_yticks(np.arange(-.5, len(names)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2); ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Validation F1 per entity label (strict match), labels sorted easiest -> hardest",
                 loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_per_label_f1.png", dpi=160)
    plt.close(fig)
    return labs, M


def fig_errors(runs, n_gold):
    kinds = [("boundary_error", "wrong boundary", S1), ("label_error", "right span, wrong label", S2),
             ("spurious", "spurious (no gold overlap)", S3), ("missed", "missed gold entity", S4)]
    names = [n for n, _, _ in EXPS][::-1]
    C = np.array([[m["errors"]["counts"].get(k, 0) for k, _, _ in kinds] for (_, m, _) in runs])[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    left = np.zeros(len(names))
    for j, (k, lab, col) in enumerate(kinds):
        ax.barh(range(len(names)), C[:, j], left=left, height=0.6, color=col, label=lab,
                edgecolor=SURFACE, linewidth=2, zorder=2)
        for i, (l0, c) in enumerate(zip(left, C[:, j])):
            if c >= 60:
                ax.text(l0 + c / 2, i, str(c), ha="center", va="center", fontsize=8, color=INK)
        left += C[:, j]
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel(f"number of errors on the 14 validation docs ({n_gold:,} gold entities)")
    ax.set_title("What kind of mistakes each model makes", loc="left", fontsize=12, color=INK)
    style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=4, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_error_types.png", dpi=160)
    plt.close(fig)
    return kinds, C[::-1]


def fig_efficiency(runs):
    pts = [(n, r["trainable_params"] / 1e6, r["total_params"] / 1e6, t)
           for (n, rel, t), (r, _, _) in zip(EXPS, runs) if t is not None]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for n, tp, tot, t in pts:
        ax.scatter(tp, t, s=60, color=S1, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        if "causal" in n:                                      # same x as the bidirectional run: one label for both
            continue
        name = "E1 full FT (bidir 0.481 / causal 0.479)" if "bidir" in n else n.split(" (")[0]
        ax.annotate(f"{name}\n{tot:.0f}M total", (tp, t), xytext=(-9 if "E1" in n else 7, 14 if "E1" in n else -2),
                    textcoords="offset points", fontsize=8, color=INK2, va="center",
                    ha="right" if "E1" in n else "left")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(0.25, 600)
    ax.set_ylim(0.33, 0.53)
    ax.set_xlabel("trainable parameters (millions, log scale)")
    ax.set_ylabel("Codabench strict macro-F1")
    ax.set_title("Score vs. number of trained parameters", loc="left", fontsize=12, color=INK)
    style(ax, "both")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_params_vs_score.png", dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- error analysis (best model)
def overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def analyse_best(texts, gold, pred, val_ids):
    rng = random.Random(0)
    out = {}
    # boundary errors by shape
    shapes, bex = Counter(), defaultdict(list)
    conf, cex = Counter(), defaultdict(list)
    spur_lab, sex = Counter(), defaultdict(list)
    miss_lab, mex = Counter(), defaultdict(list)
    for d in val_ids:
        t = texts[d]
        g = [tuple(x) for x in gold[d]]; p = [tuple(x) for x in pred[d]]
        gs = {(s, e): l for s, e, l in g}
        for s, e, l in p:
            if gs.get((s, e)) == l:
                continue
            if (s, e) in gs:
                conf[(gs[(s, e)], l)] += 1
                cex[(gs[(s, e)], l)].append(t[s:e])
                continue
            ov = [x for x in g if overlap((s, e), x)]
            if ov:
                gx = ov[0]
                if s <= gx[0] and e >= gx[1]:
                    k = "prediction too long"
                elif s >= gx[0] and e <= gx[1]:
                    k = "prediction too short"
                else:
                    k = "shifted / partial overlap"
                shapes[k] += 1
                bex[k].append((t[gx[0]:gx[1]], gx[2], t[s:e], l))
            else:
                spur_lab[l] += 1
                sex[l].append(t[max(0, s - 40):s] + "[[" + t[s:e] + "]]" + t[e:e + 40])
        for s, e, l in g:
            if not any(overlap((s, e), (ps, pe)) for ps, pe, _ in p):
                miss_lab[l] += 1
                mex[l].append(t[s:e])
    out.update(shapes=shapes, bex=bex, conf=conf, cex=cex, spur=spur_lab, sex=sex, miss=miss_lab, mex=mex)

    def inword(t, s, e):
        return (s > 0 and t[s - 1].isalnum() and t[s].isalnum()) or (e < len(t) and t[e].isalnum() and t[e - 1].isalnum())
    out["npred"] = sum(len(pred[d]) for d in val_ids)
    out["frag"] = sum(inword(texts[d], s, e) for d in val_ids for s, e, _ in pred[d])
    out["ngold"] = sum(len(gold[d]) for d in val_ids)
    out["gold_frag"] = sum(inword(texts[d], s, e) for d in val_ids for s, e, _ in gold[d])
    out["spur_auth"] = Counter(texts[d][s:e] for d in val_ids for s, e, l in pred[d]
                               if l == "AUTH" and not any(overlap((s, e), (gs, ge)) for gs, ge, _ in gold[d]))

    # position in the document and document length
    pos_hit, pos_tot = Counter(), Counter()
    per_doc = []
    for d in val_ids:
        n = len(texts[d]); P = {tuple(x) for x in pred[d]}
        for s, e, l in gold[d]:
            b = min(4, int(5 * s / n))
            pos_tot[b] += 1; pos_hit[b] += (s, e, l) in P
        sc = strict_scores({d: pred[d]}, {d: gold[d]})
        per_doc.append((d, n, len(gold[d]), sc["micro_f1"]))
    out.update(pos=[(b, pos_hit[b], pos_tot[b]) for b in range(5)], per_doc=sorted(per_doc, key=lambda x: x[1]))

    # APP/RESP: is every party mention in a doc flipped the same way?
    flips = []
    for d in val_ids:
        t = texts[d]; ps = {(s, e): l for s, e, l in pred[d]}
        c = Counter()
        for s, e, l in gold[d]:
            if l in ("APP", "RESP"):
                c[(l, ps.get((s, e), "missed/boundary"))] += 1
        if c:
            flips.append((d, dict(c)))
    out["flips"] = flips
    return out


def write_md(best, labs, M, kinds, C, runs, snap_table):
    L = []
    w = L.append
    w("# Phase 2 – Error analysis\n")
    w("Generated by `Phase2/src/analyze.py` from the kept runs' predictions on the **14 validation documents** "
      f"({best['ngold']:,} gold entities). Codabench scores are the uploaded test scores (35 documents). "
      "The detailed analysis uses the best model, **E2 LoRMA (A d768)**: Codabench 0.5090, val 0.5074.\n")
    w("Figures: `analysis/figures/`.\n")
    w("| Figure | Shows |\n|---|---|")
    w("| fig1_scores_test_vs_val.png | Codabench test macro-F1 per experiment, with local validation |")
    w("| fig2_per_label_f1.png | Validation F1 per entity label and experiment |")
    w("| fig3_error_types.png | Error types per experiment |")
    w("| fig4_params_vs_score.png | Codabench score vs. trainable parameters |\n")

    w("## 1. Scores\n")
    w("| Experiment | Trainable params | Val macro-F1 | Codabench |\n|---|---|---|---|")
    for (n, rel, t), (r, _, _) in zip(EXPS, runs):
        w(f"| {n} | {r['trainable_params']/1e6:.1f}M of {r['total_params']/1e6:.0f}M | {r['val_macro_f1']:.4f} | "
          f"{t if t is not None else 'not uploaded'} |")
    w("")

    w("## 2. Per-label F1 (validation)\n")
    w("| Experiment | " + " | ".join(labs) + " |\n|---|" + "---|" * len(labs))
    for (n, _, _), row in zip(EXPS, M):
        w(f"| {n} | " + " | ".join(f"{x:.2f}" for x in row) + " |")
    w("| **mean** | " + " | ".join(f"**{x:.2f}**" for x in M.mean(0)) + " |\n")
    w("- **Easy (F1 > 0.7):** DATE, COURT, STAT. They have regular surface forms: date patterns, "
      "\"High Court\", \"Section … of the … Act\".")
    w("- **Hard (F1 < 0.35):** APP, RESP, AUTH and WIT. APP and RESP need the *role* in the case, not the name. "
      "AUTH overlaps heavily with COURT/PREC wording. WIT appears in only 12 of 70 training documents.")
    w("- **All models agree on the three easy labels and on the hardest two (APP, RESP).** The middle labels "
      "shuffle between models (e.g. C's adapters are relatively good at WIT and bad at the counsel labels).\n")

    w("## 3. Error types\n")
    w("| Experiment | " + " | ".join(lab for _, lab, _ in kinds) + " |\n|---|" + "---|" * len(kinds))
    for (n, _, _), row in zip(EXPS, C):
        w(f"| {n} | " + " | ".join(str(x) for x in row) + " |")
    w("")
    w("- **A quarter to a third of all errors are boundary errors.** These predictions overlap a gold entity but "
      "do not match it exactly. Under strict matching, each one costs a false positive *and* a false negative.")
    w("- **LoRMA, the best model, makes the most errors in total.** It predicts many more entities (666 spurious) "
      "but misses far fewer (186). Its macro-recall is 0.577 against a macro-precision of 0.462; full "
      "fine-tuning is balanced (0.521 / 0.506). Macro-F1 weights rare labels (WIT, counsel, APP/RESP) as much "
      "as DATE, so the extra recall on rare labels outweighs the extra false positives on common ones.")
    w("- **Wrong-label errors are rare.** When a span's boundaries are exactly right, the label is almost always "
      "right too. Most of the loss comes from finding the exact extent, or from spurious and missed entities.\n")

    b = best
    w("## 4. Boundary errors (best model)\n")
    w("| Shape | Count |\n|---|---|")
    for k, v in b["shapes"].most_common():
        w(f"| {k} | {v} |")
    w("")
    for k in ("prediction too short", "prediction too long", "shifted / partial overlap"):
        ex = b["bex"][k][:6]
        if ex:
            w(f"**{k}** – gold → predicted:\n")
            for gtxt, gl, ptxt, pl in ex:
                w(f"- `{gtxt}` ({gl}) → `{ptxt}` ({pl})")
            w("")
    w("What the examples show:")
    w(f"- **Sub-word fragments (a major cause of \"too short\").** The model tags only some BPE pieces of a "
      f"word: `Th` + `ukral` for \"Thukral\", `OF 200` for \"OF 2008\", `S.` for \"S.B. Sinha\". "
      f"{b['frag']} of the best model's {b['npred']} predictions ({100*b['frag']/b['npred']:.1f}%) start or end "
      f"inside a word, against {b['gold_frag']} of {b['ngold']} gold entities. Section 10 fixes the word-internal "
      f"cases; fragments at punctuation (`S.` for \"S.B. Sinha\") remain.")
    w("- **Modifiers and prefixes (\"too long\").** `Delhi High Court` or `Division Bench of the High Court` "
      "where the gold marks only `High Court`; `O.P. No. 4612 of 1996` vs. gold `No. 4612 of 1996`.")
    w("- **Trailing punctuation.** `State Govt.` vs. gold `State Govt`; a citation followed by ` :`.")
    w("- **The annotations themselves are inconsistent** about modifiers and punctuation, which caps any "
      "strict-match score.\n")

    w("## 5. Right span, wrong label (best model)\n")
    w("| Gold → predicted | Count | Examples |\n|---|---|---|")
    for (g, p), v in b["conf"].most_common(10):
        ex = ", ".join(f"`{x}`" for x in list(dict.fromkeys(b["cex"][(g, p)]))[:3])
        w(f"| {g} → {p} | {v} | {ex} |")
    w("")
    w("- **The confusions are between related roles:** respondent vs. appellant (the most common by far), "
      "the two counsel labels, and court names that the gold marks as part of a precedent.\n")

    w("## 6. APP / RESP: the party-role problem\n")
    w("Gold APP/RESP mentions per validation document, and what the best model predicted for each:\n")
    w("| Doc | gold → predicted (count) |\n|---|---|")
    for d, c in b["flips"]:
        w(f"| {d} | " + "; ".join(f"{g}→{p}: {n}" for (g, p), n in sorted(c.items())) + " |")
    w("")
    w("Three failure modes are visible:")
    w("- **Whole-document side flip.** In 1236969, 16 of the 23 respondent mentions are labelled APP: the model "
      "found the names but put them on the wrong side.")
    w("- **Mixed sides in one document.** In 154395703, the appellant's mentions are split 9 APP / 8 RESP.")
    w("- **Not found, or wrong boundary.** In 952082, 22 of 27 appellant mentions are missed or cut. Many of "
      "these are `M/s. …` company names, where the model tags fragments such as `M` (see section 7).")
    w("- **Why roles are hard:** the role is written once, in the header (\"… Appellant Versus … "
      "Respondent\"). Mentions thousands of tokens later look the same for both sides.")
    w("- **Tried, without improvement:** prepending the first 256 header tokens to every window (Phase 2.txt, "
      "observation 11; val 0.487–0.492 vs. 0.505–0.507 without). A document-level fix, such as propagating "
      "the role decided in the header to later mentions of the same name, is the natural next step.\n")

    w("## 7. Spurious and missed entities (best model)\n")
    w("| Label | Spurious (predicted, no gold overlap) | Missed (gold, no prediction overlap) |\n|---|---|---|")
    for l in LABELS:
        w(f"| {l} | {b['spur'][l]} | {b['miss'][l]} |")
    w("")
    w("Spurious examples, with the prediction in [[ ]]:\n")
    for l, _ in b["spur"].most_common(3):
        for x in b["sex"][l][:3]:
            w(f"- {l}: `{' '.join(x.split())}`")
    w("")
    top = ", ".join(f"`{s}` {n}" for s, n in b["spur_auth"].most_common(10))
    w(f"Most frequent spurious AUTH strings: {top}.\n")
    w(f"- **AUTH is the biggest source of false positives ({b['spur']['AUTH']} of {sum(b['spur'].values())}).** "
      "Generic institutional words (`Government`, `State`, `Board`, `University`) and organisation acronyms "
      "(`AICTE`, `VESCO`) are tagged in documents where the gold does not mark them at all. Within a document "
      "the gold is consistent: every standalone repeat of an annotated string is annotated (Phase 2.txt, "
      "observation 4), so these are documents where the annotators did not treat the word as an entity.")
    w("- **Fragments add to it:** `M` from \"M/s.\" is predicted as AUTH or RESP repeatedly.")
    w("- **Some \"spurious\" predictions look like plausible entities** (`Constitution`, `Trade Marks Act, "
      "1999`) but are not marked in that document. Strict scoring counts them as errors.\n")

    w("## 8. Long documents and position\n")
    w("Strict recall by position of the gold entity in its document (best model):\n")
    w("| Position in doc | Gold entities | Recall |\n|---|---|---|")
    names = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]
    for bb, hit, tot in b["pos"]:
        w(f"| {names[bb]} | {tot} | {hit/max(1, tot):.3f} |")
    w("")
    w("Per-document micro-F1 vs. length (validation docs, shortest first):\n")
    w("| Doc | Characters | Gold entities | Micro-F1 |\n|---|---|---|---|")
    for d, n, ng, f in b["per_doc"]:
        w(f"| {d} | {n:,} | {ng} | {f:.3f} |")
    lens = np.array([x[1] for x in b["per_doc"]]); fs = np.array([x[3] for x in b["per_doc"]])
    w(f"\nCorrelation between document length and micro-F1: r = {np.corrcoef(lens, fs)[0, 1]:+.2f} "
      "(14 documents, so only indicative).\n")
    w("- **No clear position effect.** Recall stays between 0.59 and 0.69 across the document. The first 20% "
      "is not better even though it contains the header; it holds most of the hard party and counsel labels.")
    w("- **A weak length effect.** The longest documents score lower (the 114k-character document 0.505), but "
      "the worst one (952082, 0.217) is of average length: its errors are the APP/RESP problem (section 6).")
    w("- **Sliding windows cover every token.** Windows are 1,024 tokens for A/C, 768 for B and 2,048 for D, "
      "with 50% overlap. Each token keeps the prediction from the window where it is most central, so no "
      "entity is cut by a window edge.\n")

    w("## 9. Ceiling of the token pipeline\n")
    w("Converting gold spans to BIO tags and back gives macro-F1 **0.9944** (0.9875 with the word snapping from "
      "section 10, which wrongly extends the few gold entities that end inside a word, e.g. \"Railway Board\" in "
      "\"Railway Boards\"). The tokenization is not what limits the scores above.\n")

    w("## 10. Fix found by this analysis: snap predictions to word boundaries\n")
    w("Gold entities almost never start or end inside a word, but the models' predictions sometimes do "
      "(section 4). Extending every predicted span to whole words, then merging spans that now overlap, "
      "improves **every** model without retraining (validation, same predictions as above):\n")
    w("| Experiment | Val macro-F1 | + word snapping | Change |\n|---|---|---|---|")
    for n, a, s in snap_table:
        w(f"| {n} | {a:.4f} | {s:.4f} | {s - a:+.4f} |")
    w("")
    w("- **Confirmed on the test set:** the same LoRMA model scores **0.5174** on Codabench with snapping, "
      "against 0.5090 without (+0.008). This is the best Phase 2 result.")
    w("- **Snapping is now part of decoding** (`ner_data.snap_to_words`).")
    w("- **Snapped submission of the best model:** `submissions/E2_A_lorma/submission_20261009_192755_"
      "A_d768_lorma_684310_wordsnap`.\n")
    (ERR / "error_analysis.md").write_text("\n".join(L), encoding="utf-8")


def main():
    FIG.mkdir(parents=True, exist_ok=True); ERR.mkdir(parents=True, exist_ok=True)
    runs = [load_run(rel) for _, rel, _ in EXPS]
    fig_scores(runs)
    labs, M = fig_per_label(runs)
    texts, gold, _, split = load_split()
    kinds, C = fig_errors(runs, sum(len(gold[d]) for d in split["val"]))
    fig_efficiency(runs)
    vgold = {d: gold[d] for d in split["val"]}
    snap_table = [(n, strict_scores(p, vgold)["macro_f1"],
                   strict_scores({d: snap_to_words(texts[d], p[d]) for d in split["val"]}, vgold)["macro_f1"])
                  for (n, _, _), (_, _, p) in zip(EXPS, runs)]
    best_pred = runs[[rel for _, rel, _ in EXPS].index(BEST)][2]
    best = analyse_best(texts, gold, best_pred, split["val"])
    write_md(best, labs, M, kinds, C, runs, snap_table)
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))
    print("written:", ERR / "error_analysis.md")


if __name__ == "__main__":
    main()
