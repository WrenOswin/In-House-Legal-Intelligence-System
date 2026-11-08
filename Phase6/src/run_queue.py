"""Phase-6 runs, one GPU job at a time, each throttled (duty 0.6, below-normal priority).
Finished runs are skipped, so the queue can be stopped and restarted at any time:
    python run_queue.py            (S1, KD data, S5, S2-S4)      python run_queue.py S3 S4    (only these)
"TG" = teacher generation for KD (teacher_gen.py on S1's run). Required set S1-S5: all four architectures,
LoRMA + LoRA + Houlsby + Pfeiffer adapters + sequence-level KD. S6-S7 are optional extras.
"""
import subprocess
import sys
from pathlib import Path

from train import DEFAULTS, run_experiment, run_id_of

RUNS = {
    "S1": dict(exp="S1_A_lorma", backbone="A_d768", method="lorma", lr=5e-4),             # LoRMA (compulsory); KD teacher
    "S2": dict(exp="S2_D_lora", backbone="D_d560", method="lora", lr=1e-3),               # native 2048 window
    "S3": dict(exp="S3_C_houlsby", backbone="C_d560", method="houlsby", lr=1e-3),         # prefix-LM, adapters x2/block
    "S4": dict(exp="S4_B_fid_pfeiffer", backbone="B_d512", method="pfeiffer", lr=1e-3),   # FiD over 3 x 512 chunks
    "S5": dict(exp="S5_KD", backbone="A_d560", method="full", lr=1e-4, kd_teacher="S1"),  # 58M student, seq-level KD
    # optional extras
    "S6": dict(exp="S6_A_full_noKD", backbone="A_d560", method="full", lr=1e-4),          # KD baseline (same student)
    "S7": dict(exp="S7_D_lora_head", backbone="D_d560", method="lora", lr=1e-3, context="head"),  # long-doc ablation
}
DEFAULT_QUEUE = ["S1", "TG", "S5", "S2", "S3", "S4", "S6", "S7"]
WORK_RUNS = Path("C:/kaggle/working/cs781/phase6/runs")


def resolve(name):
    """Config of a queue entry with kd_teacher ("S1") replaced by the teacher's run dir."""
    cfg = dict(RUNS[name])
    if cfg.get("kd_teacher") in RUNS:
        cfg["kd_teacher"] = str(run_dir(cfg["kd_teacher"]))
    return cfg


def run_dir(name):
    cfg = {**DEFAULTS, **resolve(name)}
    return WORK_RUNS / cfg["exp"] / run_id_of(cfg)


if __name__ == "__main__":
    for name in sys.argv[1:] or DEFAULT_QUEUE:
        if name == "TG":
            t = run_dir("S1")
            if not (t / "train_gen.done").exists():
                subprocess.run([sys.executable, str(Path(__file__).with_name("teacher_gen.py")), str(t)], check=True)
            continue
        cfg = resolve(name)
        if cfg.get("kd_teacher"):
            assert (Path(cfg["kd_teacher"]) / "train_gen.done").exists(), f"{name} needs teacher summaries"
        run_experiment(cfg)
