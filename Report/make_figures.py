"""Figures for the report (vector PDF). Numbers come from the phase logs and Phase-1 history.json files."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
RUNS = r"C:\kaggle\working\cs781\runs"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
C = {"A": "#2a6fdb", "B": "#d9534f", "C": "#3aa76d", "D": "#e0a030", "grey": "#7a8ca8"}


def box(ax, x, y, w, h, text, fc="#eef3fb", ec="#2a6fdb", fs=7, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=0.8, color="#444"))


# ---------------------------------------------------------------- 1. architecture + system diagram
def fig_system():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [0.8, 1.6]})
    for a in (a1, a2):
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        a.axis("off")
    # (a) decoder block
    a1.set_title("(a) A: decoder-only block (x12)", fontsize=8)
    box(a1, 0.15, 0.02, 0.7, 0.09, "token emb. (35k BPE, tied)", fc="#f5f5f5", ec="#888")
    box(a1, 0.10, 0.17, 0.8, 0.62, "", fc="#fbfcff", ec="#2a6fdb")
    box(a1, 0.18, 0.21, 0.64, 0.08, "RMSNorm", fc="#ffffff", ec="#999")
    box(a1, 0.18, 0.33, 0.64, 0.11, "causal MHA, 12 heads\nRoPE on q,k", ec="#2a6fdb")
    box(a1, 0.18, 0.49, 0.64, 0.08, "RMSNorm", fc="#ffffff", ec="#999")
    box(a1, 0.18, 0.61, 0.64, 0.11, "SwiGLU FFN\n768 -> 3072 -> 768", ec="#2a6fdb")
    a1.text(0.93, 0.47, "residual", rotation=90, fontsize=6, va="center", color="#666")
    box(a1, 0.15, 0.85, 0.7, 0.09, "RMSNorm -> LM head (tied)", fc="#f5f5f5", ec="#888")
    arrow(a1, 0.5, 0.11, 0.5, 0.17)
    arrow(a1, 0.5, 0.79, 0.5, 0.85)
    # (b) system
    a2.set_title("(b) one Phase-1 backbone, six fine-tuned descendants", fontsize=8)
    box(a2, 0.0, 0.40, 0.23, 0.22, "Phase-1 LMs\nA 140M / 58M\nB 73M, C 58M\nD 54M", fc="#e8eefb", bold=False)
    box(a2, 0.29, 0.40, 0.2, 0.22, "adaptation\nLoRMA, LoRA, rsLoRA,\nIA3, prefix, adapters,\nBitFit, partial/full\n+ KD (140M->58M)",
        fc="#fff6e6", ec="#e0a030", fs=6)
    arrow(a2, 0.23, 0.51, 0.29, 0.51)
    heads = [
        ("P2 NER: BIO tags + constrained Viterbi", 0.88),
        ("P3 RR: sentence pool + BiGRU + CRF", 0.73),
        ("P4 CJPE: chunks + attention pool, 2-way", 0.58),
        ("P5 LSI: chunks + label attention, 100 sigmoids", 0.43),
        ("P6 Summ: decoder on source+summary / FiD", 0.28),
        ("P7 PCR: chunk maxsim bi-encoder + BM25", 0.13),
    ]
    for t, y in heads:
        box(a2, 0.54, y - 0.055, 0.44, 0.11, t, fc="#eef8f1", ec="#3aa76d", fs=5.6)
        arrow(a2, 0.49, 0.51, 0.54, y)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_system.pdf"))


# ---------------------------------------------------------------- 2. data statistics
def fig_data():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 1.95), gridspec_kw={"width_ratios": [1, 1.25]})
    ner = [("DATE", 1575), ("STAT", 1260), ("PREC", 1183), ("COURT", 1013), ("AUTH", 1003), ("CASENO", 760),
           ("APP", 407), ("RESP", 381), ("JUDGE", 245), ("A.COUNSEL", 195), ("WIT", 187), ("R.COUNSEL", 184)]
    a1.barh([n for n, _ in ner][::-1], [c for _, c in ner][::-1], color=C["A"])
    a1.set_title("(a) P2: entity mentions in 70 training docs", fontsize=8)
    a1.set_xlabel("mentions")
    roles = ["Fact", "Ratio", "PrecRelied", "ArgResp", "ArgPet", "RulLower", "Statute", "Dissent",
             "PrecNotRel", "RulPresent", "None", "Issue", "PrecOverr."]
    cl = [2979, 2856, 1305, 996, 976, 438, 367, 284, 209, 113, 53, 17, 0]
    it = [1119, 1445, 640, 352, 637, 436, 732, 0, 360, 194, 244, 90, 3]
    tot_cl, tot_it = sum(cl), sum(it)
    x = range(len(roles))
    a2.bar([i - 0.2 for i in x], [100 * v / tot_cl for v in cl], 0.4, label="CL (10,593 sent.)", color=C["A"])
    a2.bar([i + 0.2 for i in x], [100 * v / tot_it for v in it], 0.4, label="IT (6,252 sent.)", color=C["D"])
    a2.set_xticks(list(x), roles, rotation=55, ha="right", fontsize=6.5)
    a2.set_ylabel("% of sentences")
    a2.set_title("(b) P3: rhetorical-role distribution (train)", fontsize=8)
    a2.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_data.pdf"))


# ---------------------------------------------------------------- 3. learning curves
def fig_curves():
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.0, 2.55))
    runs = [("A_decoder_only/d560_L8_T1024_v32k_6613f2", "A 58M", C["A"], "--"),
            ("B_encoder_decoder/d512_E6D6_S768T256_v32k_05421b", "B 73M", C["B"], "-"),
            ("C_prefix_lm/d560_L8_T1024_v32k_6cce41", "C 58M", C["C"], "-"),
            ("D_efficient/d560_L8_T2048_v32k_05e1ef", "D 54M", C["D"], "-"),
            ("A_decoder_only/d768_L12_T1024_v35k_5fbdf6", "A 140M", C["A"], "-"),
            ("A_decoder_only_gqa/d576_L23_T1024_v35k_fbb29e", "A-GQA 132M", "#6b3fa0", ":")]
    for path, lab, col, ls in runs:
        h = json.load(open(os.path.join(RUNS, path, "history.json")))
        a1.plot([r["tokens"] / 1e6 for r in h], [r["val"] for r in h], ls, color=col, label=lab, lw=1.1)
    a1.set_xlabel("training tokens (M)")
    a1.set_ylabel("val loss (nats/token)")
    a1.set_title("(a) P1 pretraining", fontsize=8)
    a1.set_ylim(1.8, 4.0)
    a1.legend(frameon=False, fontsize=5.8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ep = [0.5, 1.0, 1.5, 2.0]
    for lab, v, col, ls in [("J1 A140M LoRMA", [0.5885, 0.7132, 0.7290, 0.7417], C["A"], "-"),
                            ("J5 A58M KD", [0.7072, 0.7410, 0.7644, 0.7735], "#0b3d91", "-"),
                            ("K1 A58M no KD", [0.6183, 0.7240, 0.7327, 0.7625], "#0b3d91", "--"),
                            ("J4 D partial", [0.5865, 0.7038, 0.7223, 0.7352], C["D"], "-"),
                            ("K2 D LoRMA", [0.5562, 0.6076, 0.6198, 0.6357], C["D"], "--"),
                            ("J2 B BitFit", [0.5483, 0.5563, 0.5541, 0.5573], C["B"], "-")]:
        a2.plot(ep, v, ls, marker="o", ms=2.5, color=col, label=lab, lw=1.1)
    a2.set_xlabel("epoch")
    a2.set_ylabel("val macro-F1")
    a2.set_title("(b) P4 CJPE", fontsize=8)
    a2.legend(frameon=False, fontsize=5.8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ep5 = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for lab, v, col, ls in [("K1 A140M LoRMA", [0.3127, 0.3345, 0.3550, 0.3635, 0.3687, 0.3769], C["A"], "-"),
                            ("K5 A58M KD", [0.3328, 0.3594, 0.3822, 0.3851, 0.3969, 0.4009], "#0b3d91", "-"),
                            ("K7 A58M no KD", [0.3075, 0.3430, 0.3678, 0.3778, 0.3844, 0.3924], "#0b3d91", "--"),
                            ("K2 D partial", [0.2904, 0.3254, 0.3456, 0.3631, 0.3719, 0.3815], C["D"], "-"),
                            ("K3 C adapters", [0.2728, 0.3012, 0.3205, 0.3306, 0.3357, 0.3435], C["C"], "-"),
                            ("K4 B full FT", [0.2774, 0.3085, 0.3304, 0.3382, 0.3408, 0.3485], C["B"], "-")]:
        a3.plot(ep5, v, ls, marker="o", ms=2.5, color=col, label=lab, lw=1.1)
    a3.set_xlabel("epoch")
    a3.set_ylabel("val macro-F1 (3k-doc subset)")
    a3.set_title("(c) P5 LSI", fontsize=8)
    a3.legend(frameon=False, fontsize=5.8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_curves.pdf"))


# ---------------------------------------------------------------- 4. cross-phase: architecture + KD
def fig_cross():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.05), gridspec_kw={"width_ratios": [1.15, 1]})
    phases = ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]
    best = {"A": [0.8281, 0.5174, 0.4759, 0.7706, 0.4033, 0.8424, 0.4312],
            "B": [0.8227, 0.3631, 0.3642, 0.5767, 0.3482, 0.8162, 0.4294],
            "C": [0.8261, 0.4019, 0.4349, 0.7586, 0.3472, 0.8309, 0.4300],
            "D": [0.8270, 0.4628, 0.4950, 0.7052, 0.3796, 0.8246, 0.4229]}
    top = [max(best[k][i] for k in best) for i in range(7)]
    mk = {"A": "o", "B": "s", "C": "^", "D": "D"}
    for k, v in best.items():
        a1.plot(range(7), [x / t for x, t in zip(v, top)], marker=mk[k], ms=4, color=C[k], lw=1.1, label=k)
    a1.set_xticks(range(7), phases)
    a1.set_ylim(0.65, 1.02)
    a1.set_ylabel("best test score / phase best")
    a1.set_title("(a) best single model per architecture", fontsize=8)
    a1.legend(ncol=4, frameon=False, fontsize=7, loc="lower right")
    a1.grid(axis="y", alpha=0.3)
    ph = ["P3", "P4", "P5", "P6", "P7"]
    teacher = [0.4146, 0.7431, 0.3770, 0.8424, 0.4312]
    student = [0.4551, 0.7706, 0.3978, 0.8346, 0.4300]
    nokd = [0.4660, 0.7618, 0.3855, 0.8379, 0.4276]
    w = 0.27
    for i in range(5):
        for j, (vals, col) in enumerate([(teacher, C["grey"]), (student, C["A"]), (nokd, "#b9c7dd")]):
            a2.bar(i + (j - 1) * w, 100 * (vals[i] / teacher[i] - 1), w, color=col,
                   label=["140M teacher", "58M KD student", "58M, no teacher"][j] if i == 0 else None)
    a2.axhline(0, color="#444", lw=0.6)
    a2.set_xticks(range(5), ph)
    a2.set_ylabel("test score vs teacher (%)")
    a2.set_title("(b) knowledge distillation", fontsize=8)
    a2.legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_cross.pdf"))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_system()
    fig_data()
    fig_curves()
    fig_cross()
    print("ok")
