"""Phase-5 runs, one GPU job at a time, each throttled (duty 0.6, below-normal priority).
Finished runs are skipped, so the queue can be stopped and restarted at any time:
    python run_queue.py            (K1-K5)      python run_queue.py K3 K4    (only these)
Required set K1-K5: all four architectures, LoRMA + 3 other techniques + KD. K6-K7 are optional extras."""
import sys
from pathlib import Path

from train import DEFAULTS, run_experiment, run_id_of

RUNS = {
    "K1": dict(exp="K1_A_lorma", backbone="A_d768", method="lorma", lr=1e-4),       # LoRMA (compulsory); KD teacher
    "K2": dict(exp="K2_D_partial", backbone="D_d560", method="partial", lr=5e-5),   # top 4 of 8 blocks
    "K3": dict(exp="K3_C_adapter", backbone="C_d560", method="adapter", bottleneck=128, lr=5e-4),
    "K4": dict(exp="K4_B_full", backbone="B_d512", method="full", lr=5e-5),         # B encoder, every weight
    "K5": dict(exp="K5_KD", backbone="A_d560", method="full", lr=3e-5, grad_ckpt=True, kd_teacher="K1"),
    # optional extras
    "K6": dict(exp="K6_KD_labelatt", backbone="A_d560", method="full", lr=3e-5, grad_ckpt=True, kd_teacher="K1",
               head="label_att"),                                                   # per-statute chunk attention
    "K7": dict(exp="K7_A_full_noKD", backbone="A_d560", method="full", lr=3e-5, grad_ckpt=True),  # KD baseline
}
DEFAULT_QUEUE = ["K1", "K2", "K3", "K4", "K5"]
WORK_RUNS = Path("C:/kaggle/working/cs781/phase5/runs")


def resolve(name):
    """Config of a queue entry with kd_teacher ("K1") replaced by the teacher's run dir."""
    cfg = dict(RUNS[name])
    if cfg.get("kd_teacher") in RUNS:
        t = cfg["kd_teacher"]
        cfg["kd_teacher"] = str(WORK_RUNS / RUNS[t]["exp"] / run_id_of({**DEFAULTS, **resolve(t)}))
    return cfg


def run_dir(name):
    cfg = {**DEFAULTS, **resolve(name)}
    return WORK_RUNS / cfg["exp"] / run_id_of(cfg)


if __name__ == "__main__":
    for name in sys.argv[1:] or DEFAULT_QUEUE:
        cfg = resolve(name)
        if cfg.get("kd_teacher"):
            assert (Path(cfg["kd_teacher"]) / "result.json").exists(), f"{name} needs the finished teacher {cfg['kd_teacher']}"
        run_experiment(cfg)
