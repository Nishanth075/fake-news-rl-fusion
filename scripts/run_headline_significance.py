from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.headline_significance import run_headline_significance
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run McNemar test for the headline RL-vs-baseline result.")
    parser.add_argument("--config", default="configs/final_headline_significance.yaml")
    return parser.parse_args()


def main() -> None:
    result = run_headline_significance(load_yaml(parse_args().config))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
