"""CLI: python run.py '{"exp": "P1_A_lorma", "backbone": "A_d768", "method": "lorma"}'"""
import json
import sys

from train import run_experiment

if __name__ == "__main__":
    for cfg in json.loads(sys.argv[1]) if sys.argv[1].lstrip().startswith("[") else [json.loads(sys.argv[1])]:
        run_experiment(cfg)
