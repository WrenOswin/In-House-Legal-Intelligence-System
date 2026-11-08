"""CLI: python run.py '{"exp": "E1_A_full", "backbone": "A_d768", "method": "full"}'  (JSON overrides of DEFAULTS)"""
import json, sys
from train_ner import run_experiment

if __name__ == "__main__":
    for arg in sys.argv[1:]:                     # several configs -> run one after another
        run_experiment(json.loads(arg))
