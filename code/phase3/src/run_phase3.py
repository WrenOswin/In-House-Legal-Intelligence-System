"""Runs the main Phase-3 experiments R1-R6 in order (R6 distils from R2) and prints a summary table.
    python run_phase3.py            (finished runs are skipped: their result.json is reused)"""
import json

from train_rr import run_experiment

EXPERIMENTS = [
    dict(exp="R1_A_lorma", backbone="A_d768", method="lorma"),
    dict(exp="R2_A_full", backbone="A_d768", method="full"),
    dict(exp="R3_B_rslora", backbone="B_d512", method="rslora"),
    dict(exp="R4_C_ia3", backbone="C_d560", method="ia3"),
    dict(exp="R5_D_prefix", backbone="D_d560", method="prefix", step_pause=0.2),
]

if __name__ == "__main__":
    results = []
    for cfg in EXPERIMENTS:
        results.append(run_experiment(cfg))
    teacher = next(r for r in results if r["exp"] == "R2_A_full")["run_dir"]
    results.append(run_experiment(dict(exp="R6_KD", backbone="A_d560", method="full", kd_teacher=teacher,
                                       step_pause=0.2)))
    print("\n=== SUMMARY (dev) ===")
    for r in results:
        print(f"{r['exp']:12s} {r['run_id']:28s} macro-F1 {r.get('dev_macro_f1_present', float('nan')):.4f} "
              f"(all-13 {r.get('dev_macro_f1_all', float('nan')):.4f}) CL {r.get('dev_CL_macro_f1', float('nan')):.4f} "
              f"IT {r.get('dev_IT_macro_f1', float('nan')):.4f} | trainable {r['trainable_params']/1e6:.2f}M | "
              f"ep {r['best_epoch']} | {r['train_minutes']} min | {r['peak_vram_gb']} GB")
        print(f"             {r.get('submission')}")
    json.dump(results, open("C:/kaggle/working/cs781/phase3/logs/phase3_main_results.json", "w", encoding="utf-8"),
              indent=1, default=str)
