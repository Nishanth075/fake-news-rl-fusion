from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.robustness import run_robustness
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled robustness experiments for RL fusion.")
    parser.add_argument("--config", default="configs/robustness.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_robustness(load_yaml(args.config))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
