from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.rl_quick_trial import run_rl_quick_trial
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fast RL-only fusion feasibility trials.")
    parser.add_argument("--config", default=None, help="Optional YAML override config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config) if args.config else None
    result = run_rl_quick_trial(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
