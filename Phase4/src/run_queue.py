"""Phase-4 runs, one GPU job at a time, each throttled (duty 0.6, below-normal priority).
Finished runs are skipped, so the queue can be stopped and restarted at any time:
    python run_queue.py            (all)      python run_queue.py J3 J4    (only these)"""
import sys

from train import run_experiment

RUNS = {
    "J1": dict(exp="J1_A_lorma", backbone="A_d768", method="lorma", lr=1e-4),        # LoRMA (compulsory); KD teacher
    "J2": dict(exp="J2_B_bitfit", backbone="B_d512", method="bitfit", lr=1e-3),
    "J3": dict(exp="J3_C_adapter", backbone="C_d560", method="adapter", lr=2e-4),
    "J4": dict(exp="J4_D_partial", backbone="D_d560", method="partial", lr=5e-5),    # top 4 of 8 blocks
    "J5": dict(exp="J5_KD", backbone="A_d560", method="full", lr=3e-5, grad_ckpt=True,
               kd_teacher="C:/kaggle/working/cs781/phase4/runs/J1_A_lorma/{J1}"),   # student of J1
    # round 2 (2026-10-20, after the TAs' dev.json fix; train/test unchanged)
    "K1": dict(exp="K1_A_full_noKD", backbone="A_d560", method="full", lr=3e-5, grad_ckpt=True),  # KD baseline
    "K2": dict(exp="K2_D_lorma", backbone="D_d560", method="lorma", lr=1e-4),       # method control vs J4
    "K3": dict(exp="K3_C_full_KD", backbone="C_d560", method="full", lr=3e-5, grad_ckpt=True,
               kd_teacher="J1"),                                                    # KD into another arch
    "K4": dict(exp="K4_B_lorma_KD", backbone="B_d512", method="lorma", lr=2e-4,
               kd_teacher="J1"),                                                    # stronger method for B
}

if __name__ == "__main__":
    done = {}
    for name in sys.argv[1:] or list(RUNS):
        cfg = dict(RUNS[name])
        if cfg.get("kd_teacher"):
            from pathlib import Path
            from train import run_id_of, DEFAULTS
            j1 = Path("C:/kaggle/working/cs781/phase4/runs/J1_A_lorma") / run_id_of({**DEFAULTS, **RUNS["J1"]})
            assert (j1 / "result.json").exists(), f"J5 needs the finished J1 teacher {j1}"
            cfg["kd_teacher"] = str(j1)
        done[name] = run_experiment(cfg)
