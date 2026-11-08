"""
Report material for Phase 3 (Rhetorical Role Prediction): figures (Phase3/analysis/figures/*.png), a results
table (analysis/results.md + .csv) and an error analysis (analysis/error_analysis/error_analysis.md), built
from the saved runs (dev predictions / metrics) and the Codabench scores recorded in Phase 3.txt.

    python analyze.py
"""
import csv
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from ensemble import average, to_labels
from evaluate import confusion, rr_scores, scores_from_confusion
from rr_data import DOMAINS, LABELS, N_LABELS, PROJECT_DIR, WORK, label_counts, load_rr

RUNS = WORK / "runs"
FIG = PROJECT_DIR / "analysis" / "figures"
ERR = PROJECT_DIR / "analysis" / "error_analysis"
OUT = PROJECT_DIR / "analysis"

# every Codabench submission: (name, run dir relative to RUNS or None for an ensemble, test macro-F1, round, note)
SUBMISSIONS = [
    ("R1 A LoRMA", "R1_A_lorma/A_d768_lorma_w_ba1538", 0.4502, 1, "A d768 140M, LoRMA+ r=8"),
    ("R2 A full FT", "R2_A_full/A_d768_full_w_4ef131", 0.4146, 1, "A d768 140M, full fine-tuning (KD teacher)"),
    ("R3 B rsLoRA", "R3_B_rslora/B_d512_rslora_w_2d2f72", 0.3642, 1, "B encoder 42M, rsLoRA r=8"),
    ("R4 C IA3", "R4_C_ia3/C_d560_ia3_w_eb3e89", 0.4349, 1, "C prefix-LM 58M, IA3"),
    ("R5 D prefix", "R5_D_prefix/D_d560_prefix_w_dca8c6", 0.4950, 1, "D efficient 54M, prefix tuning, 2048 window"),
    ("R6 KD student", "R6_KD/A_d560_full_w_kd_86811a", 0.4551, 1, "A d560 58M, full FT + KD from R2"),
    ("R7 D LoRMA 2048", "R7_D_lorma/D_d560_lorma_w_c8f1dd", 0.4441, 2, "D, LoRMA, 2048 window"),
    ("R7 D LoRMA 1024", "R7_D_lorma/D_d560_lorma_w_5232ad", 0.4536, 2, "D, LoRMA, 1024 window"),
    ("R8 D prefix, long crops", "R8_D_prefix_long/D_d560_prefix_w_b8d31c", 0.4218, 2, "R5 with 128-sentence crops"),
    ("R1b A LoRMA, 20 ep", "R1_A_lorma/A_d768_lorma_w_bc8045", 0.4743, 2, "R1 with a 20-epoch schedule"),
    ("R6 student, no KD", "R6_KD/A_d560_full_w_c030bd", 0.4660, 2, "KD baseline: same student without the teacher"),
    ("Ensemble: 4 D runs", None, 0.4306, 2, "R5 + R7 x2 + R8"),
    ("Ensemble: 10 runs", None, 0.4744, 2, "all held-out runs except the KD baseline"),
    ("R5 on train+dev", "R5_D_prefix/D_d560_prefix_w_all_bc6a76", 0.3976, 3, "full data, 15 epochs"),
    ("R1 on train+dev", "R1_A_lorma/A_d768_lorma_w_all_aea4b0", 0.4678, 3, "full data, 15 epochs"),
    ("R6 student on train+dev", "R6_KD/A_d560_full_w_all_324fae", 0.4230, 3, "full data, no KD, 15 epochs"),
    ("Ensemble: 3 train+dev runs", None, 0.4529, 3, ""),
    ("Ensemble: 10 + 3 train+dev", None, 0.4697, 3, ""),
    ("Ensemble: 10 runs, tau 0.5", None, 0.4548, 4, "logit adjustment"),
    ("Ensemble: 13 runs, tau 0.5", None, 0.4620, 4, "logit adjustment"),
    ("R1 train+dev, tau 0.5", None, 0.4759, 4, "logit adjustment"),
    ("R5, tau 0.5", None, 0.4808, 4, "logit adjustment, argmax decoding"),
]
MAIN = SUBMISSIONS[:6]                                    # the required experiments, one per technique
ENSEMBLE_DEV = {"Ensemble: 4 D runs": 0.4447, "Ensemble: 10 runs": 0.4748, "Ensemble: 10 runs, tau 0.5": 0.4836,
                "R5, tau 0.5": 0.4001}                   # dev all-13 printed by ensemble.py

# reference palette (dataviz skill, light mode) - same as Phase 2's figures
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
         "#1c5cab", "#184f95", "#104281", "#0d366b"]
SHORT = {"Fact": "Fact", "Issue": "Issue", "ArgumentPetitioner": "ArgPet", "ArgumentRespondent": "ArgResp",
         "Statute": "Statute", "Dissent": "Dissent", "PrecedentReliedUpon": "PrecRel",
         "PrecedentNotReliedUpon": "PrecNotRel", "PrecedentOverruled": "PrecOver", "RulingByLowerCourt": "RulLower",
         "RatioOfTheDecision": "Ratio", "RulingByPresentCourt": "RulPresent", "None": "None"}

plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2, "text.color": INK,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def style(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def run_dir(rel):
    return RUNS / rel


def result(rel):
    return json.load(open(run_dir(rel) / "result.json", encoding="utf-8"))


def dev_metrics(rel):
    return json.load(open(run_dir(rel) / "dev_metrics.json", encoding="utf-8"))


def dev_score(name, rel):
    if rel and (run_dir(rel) / "dev_metrics.json").exists():
        return result(rel)["dev_macro_f1_all"]
    return ENSEMBLE_DEV.get(name)


# ----------------------------------------------------------------------------- figures
def fig_scores():
    rows = [(n, dev_score(n, rel), t, rnd) for n, rel, t, rnd, _ in SUBMISSIONS][::-1]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    ax.barh(y, [t for _, _, t, _ in rows], height=0.6, color=S1, label="Codabench test (10 docs)", zorder=2)
    dv = [(yi, d) for yi, (_, d, _, _) in zip(y, rows) if d is not None]
    ax.scatter([d for _, d in dv], [yi for yi, _ in dv], s=46, color=S2, edgecolor=SURFACE, linewidth=1.5,
               zorder=3, label="local dev, all-13 macro-F1 (10 docs)")
    for yi, (_, d, t, _) in zip(y, rows):
        ax.text(max(t, d or 0) + 0.006, yi, f"{t:.3f}", va="center", color=INK, fontsize=8.5)
    ax.set_yticks(y, [n for n, _, _, _ in rows], fontsize=9)
    for yi, (_, _, _, rnd) in zip(y, rows):
        if yi > 0 and rows[yi - 1][3] != rnd:
            ax.axhline(yi - 0.5, color=AXIS, linewidth=0.8)
    ax.set_xlim(0, 0.56)
    ax.set_xlabel("macro-F1")
    ax.set_title("Phase 3 RR: all 22 submissions, by round (top = round 1)", loc="left", fontsize=12, color=INK)
    style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.07), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_scores_test_vs_dev.png", dpi=160)
    plt.close(fig)


def fig_per_label():
    names = [n for n, *_ in MAIN] + ["Ensemble: 10 runs"]
    M = [[dev_metrics(rel)["per_label"][l]["f1"] for l in LABELS] for _, rel, *_ in MAIN]
    M.append([ENS_SC["per_label"][l]["f1"] for l in LABELS])
    M = np.array(M)
    keep = [i for i, l in enumerate(LABELS) if ENS_SC["per_label"][l]["support"] > 0]   # labels present in dev
    order = sorted(keep, key=lambda i: -M[:, i].mean())
    M = M[:, order]; labs = [SHORT[LABELS[i]] for i in order]
    cmap = LinearSegmentedColormap.from_list("blue", BLUES)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if M[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(labs)), labs, fontsize=8.5)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    ax.set_xticks(np.arange(-.5, len(labs)), minor=True); ax.set_yticks(np.arange(-.5, len(names)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2); ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Dev F1 per rhetorical role (11 roles present in dev), easiest -> hardest",
                 loc="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_per_label_f1.png", dpi=160)
    plt.close(fig)


def fig_confusion():
    cm = np.array(ENS_SC["confusion"], dtype=float)
    keep = [i for i in range(N_LABELS) if cm[i].sum() > 0 or cm[:, i].sum() > 0]
    cm = cm[np.ix_(keep, keep)]
    R = cm / cm.sum(1, keepdims=True)
    labs = [SHORT[LABELS[i]] for i in keep]
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd"] + BLUES)
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    ax.imshow(R, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(keep)):
        for j in range(len(keep)):
            if cm[i, j] > 0:
                ax.text(j, i, f"{R[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=6.8,
                        color="#ffffff" if R[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(labs)), labs, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(labs)), labs, fontsize=8.5)
    ax.set_xlabel("predicted role"); ax.set_ylabel("gold role")
    ax.set_xticks(np.arange(-.5, len(labs)), minor=True); ax.set_yticks(np.arange(-.5, len(labs)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2); ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Confusion on dev, 10-run ensemble (share of each gold role, count in brackets)",
                 loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_confusion_matrix.png", dpi=160)
    plt.close(fig)


def fig_domains():
    names = [n for n, *_ in MAIN] + ["Ensemble: 10 runs"]
    cl, it = [], []                                       # all-13 per domain (result.json only has "present")
    for _, rel, *_ in MAIN:
        m = DOMAIN_ALL13[rel]
        cl.append(m["CL"]); it.append(m["IT"])
    cl.append(ENS_SC["CL"]["macro_f1_all"]); it.append(ENS_SC["IT"]["macro_f1_all"])
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    h = 0.36
    ax.barh(y + h / 2, cl, height=h, color=S1, label="Corporate Law (5 dev docs)", zorder=2,
            edgecolor=SURFACE, linewidth=1)
    ax.barh(y - h / 2, it, height=h, color=S2, label="IT (5 dev docs)", zorder=2, edgecolor=SURFACE, linewidth=1)
    for yi, a, b in zip(y, cl, it):
        ax.text(a + 0.005, yi + h / 2, f"{a:.3f}", va="center", fontsize=8, color=INK2)
        ax.text(b + 0.005, yi - h / 2, f"{b:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, names, fontsize=9)
    ax.set_xlim(0, 0.62)
    ax.set_xlabel("dev macro-F1 over all 13 roles")
    ax.set_title("Corporate Law vs IT on dev", loc="left", fontsize=12, color=INK)
    style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_cl_vs_it.png", dpi=160)
    plt.close(fig)


def fig_efficiency():
    pts = [(n, result(rel)["trainable_params"] / 1e6, result(rel)["head_params"] / 1e6,
            result(rel)["total_params"] / 1e6, t) for n, rel, t, rnd, _ in SUBMISSIONS[:6] + [SUBMISSIONS[10]]]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    offsets = {"R1 A LoRMA": (8, 6), "R2 A full FT": (-8, -12), "R3 B rsLoRA": (8, 0), "R4 C IA3": (8, -8),
               "R5 D prefix": (8, 4), "R6 KD student": (-8, -12), "R6 student, no KD": (-8, 10)}
    for n, tp, hp, tot, t in pts:
        ax.scatter(tp, t, s=60, color=S1, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        dx, dy = offsets[n]
        ax.annotate(f"{n}\n{tot:.0f}M total", (tp, t), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                    color=INK2, va="center", ha="left" if dx > 0 else "right")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(1, 600)
    ax.set_ylim(0.34, 0.52)
    ax.set_xlabel("trainable parameters incl. the 2.4-2.8M BiGRU/CRF head (millions, log scale)")
    ax.set_ylabel("Codabench macro-F1")
    ax.set_title("Score vs. number of trained parameters", loc="left", fontsize=12, color=INK)
    style(ax, "both")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_params_vs_score.png", dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- tables
def write_results():
    rows = []
    for n, rel, t, rnd, note in SUBMISSIONS:
        r = result(rel) if rel else {}
        rows.append(dict(round=rnd, submission=n, note=note, run_id=r.get("run_id", ""), codabench=t,
                         dev_macro_f1_all=dev_score(n, rel), dev_macro_f1_present=r.get("dev_macro_f1_present"),
                         trainable_params_M=round(r["trainable_params"] / 1e6, 3) if r else None,
                         head_params_M=round(r["head_params"] / 1e6, 3) if r else None,
                         best_epoch=r.get("best_epoch"), train_minutes=r.get("train_minutes"),
                         peak_vram_gb=r.get("peak_vram_gb"), dev_infer_sec=r.get("dev_infer_sec")))
    with open(OUT / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    fmt = lambda v, d=4: "-" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))
    lines = ["# Phase 3 – Rhetorical Role Prediction: results", "",
             "Generated by `Phase3/src/analyze.py`. Codabench = macro-F1 on the 10 hidden test documents. "
             "Dev = local macro-F1 on the 10 dev documents, averaged over all 13 roles (the variant that tracked "
             "Codabench; the 'present' variant averages only roles occurring in dev and is ~0.08 higher).", "",
             "| Round | Submission | Codabench | Dev all-13 | Dev present | Trainable (M) | Best ep | Train min | Peak VRAM GB |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        best = " **(best)**" if r["codabench"] == max(x["codabench"] for x in rows) else ""
        lines.append(f"| {r['round']} | {r['submission']}{best} | {fmt(r['codabench'])} | {fmt(r['dev_macro_f1_all'])} | "
                     f"{fmt(r['dev_macro_f1_present'])} | {fmt(r['trainable_params_M'], 2)} | {fmt(r['best_epoch'])} | "
                     f"{fmt(r['train_minutes'], 1)} | {fmt(r['peak_vram_gb'], 2)} |")
    d = np.array([r["dev_macro_f1_all"] for r in rows if r["dev_macro_f1_all"] is not None])
    t = np.array([r["codabench"] for r in rows if r["dev_macro_f1_all"] is not None])
    lines += ["", f"Over the {len(d)} submissions with a dev score: test − dev = {np.mean(t-d):+.3f} ± {np.std(t-d):.3f}; "
              f"Pearson r = {np.corrcoef(d, t)[0, 1]:.2f}.", "",
              "Figures: `analysis/figures/fig1`–`fig5`. Error analysis: `analysis/error_analysis/error_analysis.md`."]
    open(OUT / "results.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return rows


def write_error_analysis(data):
    dev = data["dev"]
    gold = {d.id: [LABELS[i] for i in d.labels] for d in dev}
    pred = ENS_PRED
    per = ENS_SC["per_label"]
    tr = label_counts(data["train"])
    L = ["# Phase 3 – Error analysis", "",
         "Generated by `Phase3/src/analyze.py` from the dev predictions (10 documents, 2,142 sentences). The detailed "
         "analysis uses the **10-run ensemble** (averaged CRF marginals of every held-out run except the KD baseline): "
         f"dev macro-F1 {ENS_SC['macro_f1_all']:.4f} (all 13 roles) / {ENS_SC['macro_f1_present']:.4f} (roles present), "
         "Codabench 0.4744. It is more stable than any single run, so its errors are the systematic ones.", "",
         "| Figure | Shows |", "|---|---|",
         "| fig1_scores_test_vs_dev.png | All 22 Codabench scores with the local dev score |",
         "| fig2_per_label_f1.png | Dev F1 per role for the six main experiments and the ensemble |",
         "| fig3_confusion_matrix.png | Row-normalised confusion matrix of the ensemble on dev |",
         "| fig4_cl_vs_it.png | Dev macro-F1 per domain |",
         "| fig5_params_vs_score.png | Codabench score vs trainable parameters |", "",
         "## 1. Per role (ensemble, dev)", "",
         "| Role | Train count | Dev gold | Predicted | P | R | F1 |", "|---|---|---|---|---|---|---|"]
    for i, l in enumerate(LABELS):
        v = per[l]
        L.append(f"| {l} | {tr[i]:,} | {v['support']} | {v['n_pred']} | {v['p']:.3f} | {v['r']:.3f} | {v['f1']:.3f} |")
    cm = np.array(ENS_SC["confusion"])
    off = cm.copy(); np.fill_diagonal(off, 0)
    top = np.dstack(np.unravel_index(np.argsort(-off.ravel())[:10], off.shape))[0]
    L += ["", f"Dissent and PrecedentOverruled do not occur in dev and are never predicted; under the all-13 average "
          f"they count as F1 = 0, which caps the all-13 score at 11/13 = 0.846.", "",
          "## 2. Most frequent confusions (gold → predicted)", "",
          "| Gold | Predicted | Sentences | Share of gold role |", "|---|---|---|---|"]
    for a, b in top:
        L.append(f"| {LABELS[a]} | {LABELS[b]} | {cm[a, b]} | {cm[a, b] / cm[a].sum():.2f} |")
    # per domain
    L += ["", "## 3. Corporate Law vs IT (ensemble, dev, F1 per role)", "", "| Role | CL gold | CL F1 | IT gold | IT F1 |",
          "|---|---|---|---|---|"]
    dom_sc = {}
    for di, dom in enumerate(DOMAINS):
        g = {d.id: gold[d.id] for d in dev if d.domain == di}
        dom_sc[dom] = scores_from_confusion(confusion(pred, g))
    for l in LABELS:
        a, b = dom_sc["CL"]["per_label"][l], dom_sc["IT"]["per_label"][l]
        if a["support"] or b["support"]:
            L.append(f"| {l} | {a['support']} | {a['f1']:.3f} | {b['support']} | {b['f1']:.3f} |")
    L.append(f"| **macro (all 13)** | | **{dom_sc['CL']['macro_f1_all']:.3f}** | | **{dom_sc['IT']['macro_f1_all']:.3f}** |")
    # run lengths: where do errors sit?
    at_b, in_r, nb, nr = 0, 0, 0, 0
    for d in dev:
        y, p = d.labels, [LABELS.index(x) for x in pred[d.id]]
        for t in range(len(y)):
            boundary = (t > 0 and y[t] != y[t - 1]) or (t + 1 < len(y) and y[t] != y[t + 1])
            if boundary:
                nb += 1; at_b += p[t] != y[t]
            else:
                nr += 1; in_r += p[t] != y[t]
    pred_switch = sum(sum(a != b for a, b in zip(pred[d.id], pred[d.id][1:])) for d in dev)
    gold_switch = sum(int((d.labels[1:] != d.labels[:-1]).sum()) for d in dev)
    L += ["", "## 4. Errors at role boundaries vs inside runs", "",
          f"- Sentences next to a gold role change: {nb}, error rate {at_b / nb:.2f}.",
          f"- Sentences inside a run of one role: {nr}, error rate {in_r / nr:.2f}.",
          f"- Role changes in the gold sequences: {gold_switch}; in the predicted sequences: {pred_switch}.", ""]
    L += ["## 5. Findings", "",
          "1. **The models fall back on the majority role.** RatioOfTheDecision is predicted "
          f"{per['RatioOfTheDecision']['n_pred']} times for {per['RatioOfTheDecision']['support']} gold sentences; "
          "Fact, precedents, arguments and lower-court rulings are all pulled into it. The rarer roles are "
          f"under-predicted: RulingByLowerCourt {per['RulingByLowerCourt']['n_pred']}/{per['RulingByLowerCourt']['support']}, "
          f"None {per['None']['n_pred']}/{per['None']['support']}, Issue {per['Issue']['n_pred']}/{per['Issue']['support']}. "
          "Macro-F1 weights these rare roles as much as Ratio, so this is where the score is lost.",
          "2. **Closely related roles are the hardest pairs:** ArgumentPetitioner vs ArgumentRespondent (the "
          "wording of the two sides' arguments is the same; only who is arguing differs), PrecedentReliedUpon vs "
          "Fact/Ratio (a cited case is reasoning or background depending on how the court uses it), and "
          "RulingByLowerCourt vs Ratio/Fact (a lower court's decision is described in the same language as the "
          "present court's).",
          "3. **Whole segments get the wrong role.** The error rate is higher next to a gold role change "
          f"({at_b / nb:.2f}) than inside a run ({in_r / nr:.2f}), but most wrong sentences ({in_r} of {in_r + at_b}) "
          f"sit inside runs, and the predicted sequences change role about as often as the gold ones "
          f"({pred_switch} vs {gold_switch}): the BiGRU/CRF finds the segments but labels whole segments with the "
          "wrong role. The 'previous gold role' oracle reaches 0.638 all-13 macro-F1 on dev, more than any model "
          "(best single run 0.479).",
          "4. **Domain:** IT scores higher than Corporate Law for 4 of the 6 main models on dev (R2 is better on "
          "CL, R3 ties; fig 4), and for the ensemble (0.475 vs 0.399). Part of the gap is arithmetic: the CL dev "
          "documents contain only 10 of the 13 roles (no PrecedentNotReliedUpon), so their all-13 average is capped "
          "lower, and CL has only 3 Issue sentences (F1 0). CL documents are also longer (dev median 184 vs 120 "
          "sentences) and dominated by long argument sections where the two sides alternate (ArgPet/ArgResp).",
          "5. **Noise:** 10 dev and 10 test documents. Codabench − dev = +0.009 ± 0.037 over the 15 submissions with a dev score (Pearson r 0.37), "
          "submissions, and the same configuration trained on 10 more documents moved by −0.10 to +0.02. "
          "Differences below ~0.05 between systems are not resolvable; the robust findings are the architecture "
          "gap (A/D/C vs B) and that parameter-efficient methods match full fine-tuning (LoRMA 0.450–0.476 vs "
          "full FT 0.415 on Codabench), both also seen in Phase 2.",
          "6. **Logit adjustment** (subtracting 0.5·log prior to favour rare roles) gave +0.01 on dev but −0.009 on "
          "average on Codabench (4 submissions): it does not reliably fix finding 1."]
    open(ERR / "error_analysis.md", "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True); ERR.mkdir(parents=True, exist_ok=True)
    data = load_rr()
    dev = data["dev"]
    gold = {d.id: [LABELS[i] for i in d.labels] for d in dev}
    domain_of = {d.id: DOMAINS[d.domain] for d in dev}
    held = sorted(os.path.normpath(p) for p in glob.glob(str(RUNS) + "/R*/*/")
                  if "c030bd" not in p and "_all_" not in p)
    assert len(held) == 10, held
    ENS_PRED = to_labels(average(held, "dev"))
    ENS_SC = rr_scores(ENS_PRED, gold, domain_of)
    for dom in DOMAINS:                                   # all-13 variant per domain for the ensemble
        g = {k: v for k, v in gold.items() if domain_of[k] == dom}
        ENS_SC[dom]["macro_f1_all"] = scores_from_confusion(confusion(ENS_PRED, g))["macro_f1_all"]
    DOMAIN_ALL13 = {}
    for _, rel, *_ in MAIN:
        p = json.load(open(run_dir(rel) / "dev_predictions.json", encoding="utf-8"))
        DOMAIN_ALL13[rel] = {dom: scores_from_confusion(confusion(p, {k: v for k, v in gold.items()
                                                                   if domain_of[k] == dom}))["macro_f1_all"]
                             for dom in DOMAINS}
    fig_scores(); fig_per_label(); fig_confusion(); fig_domains(); fig_efficiency()
    write_results(); write_error_analysis(data)
    print("written:", sorted(p.name for p in FIG.glob("*.png")), "results.md/.csv, error_analysis.md")
