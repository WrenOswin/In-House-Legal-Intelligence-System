"""CLI: python run.py '{"exp": "R1_A_lorma", "backbone": "A_d768", "method": "lorma"}'  (JSON overrides of DEFAULTS)"""
import json, sys
from train_rr import run_experiment

if __name__ == "__main__":
    for arg in sys.argv[1:]:                     # several configs -> run one after another
        run_experiment(json.loads(arg))
