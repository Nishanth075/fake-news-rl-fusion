from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.seed_significance import run_seed_significance
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-seed RL fusion comparison with McNemar tests.")
    parser.add_argument("--config", default="configs/final_seed_significance.yaml")
    return parser.parse_args()


def main() -> None:
    result = run_seed_significance(load_yaml(parse_args().config))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
