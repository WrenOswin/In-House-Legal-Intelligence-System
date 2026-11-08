"""Phase-3 round 3 (2026-10-12): retrain the three best test-set configurations on train+dev (90 docs) with
the SAME 15-epoch schedule they were selected with (R1/R5 peaked at epochs 15/13 of 15, so the final
model of a full-data run matches the model that was kept), then ensemble.
    python run_round3.py            (finished runs are skipped)"""
import glob
import os

from ensemble import run as ensemble
from train_rr import run_experiment

RUNS = "C:/kaggle/working/cs781/phase3/runs/"
EXPERIMENTS = [
    dict(exp="R5_D_prefix", backbone="D_d560", method="prefix", full_data=True, epochs=15),   # test 0.4950
    dict(exp="R1_A_lorma", backbone="A_d768", method="lorma", full_data=True, epochs=15),    # test 0.4502 / 0.4743
    dict(exp="R6_KD", backbone="A_d560", method="full", full_data=True, epochs=15),          # no teacher, test 0.4660
]

if __name__ == "__main__":
    full = []
    for cfg in EXPERIMENTS:
        try:
            full.append(run_experiment(cfg)["run_dir"])
        except Exception as e:
            print(f"FAILED {cfg}: {e!r}", flush=True)
    ensemble("full3_ensemble", full)
    held_out = sorted(os.path.normpath(p) for p in glob.glob(RUNS + "R*/*/")
                      if "c030bd" not in p and "_all_" not in p)          # the 10 runs of all10_ensemble
    ensemble("all10_full3_ensemble", held_out + full)
