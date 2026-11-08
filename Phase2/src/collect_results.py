"""Collect every finished Phase-2 run into Phase2/analysis/results.csv and results.md (one row per run)."""
import csv
import glob
import json
from pathlib import Path

from ner_data import LABELS, PROJECT_DIR

RUNS = "C:/kaggle/working/cs781/phase2/runs/*/*/result.json"
COLS = ["exp", "run_id", "backbone", "method", "bidirectional", "kd", "val_macro_f1", "val_macro_p",
        "val_macro_r", "val_micro_f1", "best_epoch", "total_params_M", "trainable_params_M", "trainable_pct",
        "peak_vram_gb", "train_minutes", "val_infer_sec", "submission"]


def collect():
    rows = []
    for p in sorted(glob.glob(RUNS)):
        r = json.load(open(p, encoding="utf-8"))
        cfg = json.load(open(Path(p).parent / "config.json", encoding="utf-8"))
        row = {k: r.get(k) for k in COLS}
        row.update(kd=bool(cfg.get("kd_teacher")),
                   total_params_M=round(r["total_params"] / 1e6, 2), trainable_params_M=round(r["trainable_params"] / 1e6, 3),
                   trainable_pct=round(100 * r["trainable_params"] / r["total_params"], 2),
                   submission=Path(r["submission"]).parent.name if r.get("submission") else "")
        vm = Path(p).parent / "val_metrics.json"
        if vm.exists():
            m = json.load(open(vm, encoding="utf-8"))
            for l in LABELS:
                row[f"f1_{l}"] = round(m["scores"]["per_label"][l]["f1"], 3)
            row.update({f"err_{k}": v for k, v in m["errors"]["counts"].items()})
        for k in ("val_macro_f1", "val_macro_p", "val_macro_r", "val_micro_f1"):
            if row[k] is not None:
                row[k] = round(row[k], 4)
        rows.append(row)
    return rows


if __name__ == "__main__":
    rows = collect()
    out = PROJECT_DIR / "analysis"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    show = ["exp", "backbone", "method", "bidirectional", "val_macro_f1", "val_micro_f1", "best_epoch",
            "trainable_params_M", "trainable_pct", "peak_vram_gb", "train_minutes"]
    lines = ["| " + " | ".join(show) + " |", "|" + "---|" * len(show)]
    lines += ["| " + " | ".join(str(r.get(k, "")) for k in show) + " |" for r in rows]
    per = ["exp", "backbone", "method"] + [f"f1_{l}" for l in LABELS]
    lines += ["", "Per-label validation F1", "", "| " + " | ".join(per) + " |", "|" + "---|" * len(per)]
    lines += ["| " + " | ".join(str(r.get(k, "")) for k in per) + " |" for r in rows if "f1_DATE" in r]
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
