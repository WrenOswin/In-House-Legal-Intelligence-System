"""python run.py '{"exp": "S1_A_lorma", "backbone": "A_d768", "method": "lorma"}'   (one run from a JSON config)"""
import json
import sys

from train import run_experiment

if __name__ == "__main__":
    run_experiment(json.loads(sys.argv[1]))
