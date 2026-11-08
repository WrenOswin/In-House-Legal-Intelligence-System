"""Phase-3 round 2 (2026-10-12, after the first Codabench scores): context-length and schedule experiments.
Model selection on dev macro-F1 over all 13 labels (tracked Codabench best, Phase 3.txt obs. 9).
    python run_round2.py            (finished runs are skipped)"""
import json

from train_rr import run_experiment

COMMON = dict(select_metric="macro_f1_all", epochs=20, patience=5)
EXPERIMENTS = [
    dict(exp="R7_D_lorma", backbone="D_d560", method="lorma"),                       # D with A's best method
    dict(exp="R7_D_lorma", backbone="D_d560", method="lorma", max_len=1024),         # same, half the context
    dict(exp="R8_D_prefix_long", backbone="D_d560", method="prefix", chunk_sents=128, chunks_per_step=2),
    dict(exp="R1_A_lorma", backbone="A_d768", method="lorma"),                        # R1 with 20 epochs
    dict(exp="R6_KD", backbone="A_d560", method="full"),                              # KD baseline: no teacher
]

if __name__ == "__main__":
    results = []
    for cfg in EXPERIMENTS:
        try:
            results.append(run_experiment({**COMMON, **cfg}))
        except Exception as e:                       # keep going: one failed run must not stop the rest
            print(f"FAILED {cfg}: {e!r}", flush=True)
    print("\n=== SUMMARY (dev) ===")
    for r in results:
        print(f"{r['exp']:17s} {r['run_id']:30s} all-13 {r['dev_macro_f1_all']:.4f} present {r['dev_macro_f1_present']:.4f}"
              f" | ep {r['best_epoch']} | {r['train_minutes']} min | {r['peak_vram_gb']} GB\n   {r.get('submission')}")
    json.dump(results, open("C:/kaggle/working/cs781/phase3/logs/round2_results.json", "w", encoding="utf-8"),
              indent=1, default=str)
