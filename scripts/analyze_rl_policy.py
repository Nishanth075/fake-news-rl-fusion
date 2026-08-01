from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.analysis import analyze_rl_policy
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze trained RL fusion policy behavior.")
    parser.add_argument("--config", default="configs/rl_analysis.yaml")
    return parser.parse_args()


def main() -> None:
    analysis = analyze_rl_policy(load_yaml(parse_args().config))
    print(json.dumps(analysis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
