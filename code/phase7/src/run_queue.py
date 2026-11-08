"""Phase-7 runs, one GPU job at a time, each throttled (below-normal priority).
Finished runs are skipped, so the queue can be stopped and restarted at any time:
    python run_queue.py            (P1-P6)      python run_queue.py P3 P4    (only these)
Required set: all four architectures, LoRMA + partial FT + IA3 + full FT + KD (+ the KD baseline P6)."""
import sys
from pathlib import Path

from train import DEFAULTS, run_experiment, run_id_of

RUNS = {
    "P1": dict(exp="P1_A_lorma", backbone="A_d768", method="lorma"),                 # LoRMA (compulsory); KD teacher
    "P2": dict(exp="P2_D_partial", backbone="D_d560", method="partial"),             # top 4 of 8 blocks
    "P3": dict(exp="P3_C_ia3", backbone="C_d560", method="ia3"),                     # IA3 rescaling vectors
    "P4": dict(exp="P4_B_full", backbone="B_d512", method="full"),                   # B encoder, every weight
    "P5": dict(exp="P5_KD", backbone="A_d560", method="full", grad_ckpt=True, kd_teacher="P1"),
    "P6": dict(exp="P6_A_full_noKD", backbone="A_d560", method="full", grad_ckpt=True),   # KD baseline
}
DEFAULT_QUEUE = ["P6", "P1", "P2", "P3", "P4", "P5"]
WORK_RUNS = Path("C:/kaggle/working/cs781/phase7/runs")


def resolve(name):
    """Config of a queue entry with kd_teacher ("P1") replaced by the teacher's run dir."""
    cfg = dict(RUNS[name])
    if cfg.get("kd_teacher") in RUNS:
        t = cfg["kd_teacher"]
        full = {**DEFAULTS, **resolve(t)}
        if full["lr"] is None:
            from train import DEFAULT_LR
            full["lr"] = DEFAULT_LR[full["method"]]
        cfg["kd_teacher"] = str(WORK_RUNS / RUNS[t]["exp"] / run_id_of(full))
    return cfg


if __name__ == "__main__":
    for name in sys.argv[1:] or DEFAULT_QUEUE:
        cfg = resolve(name)
        if cfg.get("kd_teacher"):
            assert (Path(cfg["kd_teacher"]) / "result.json").exists(), f"{name} needs the finished teacher {cfg['kd_teacher']}"
        run_experiment(cfg)
