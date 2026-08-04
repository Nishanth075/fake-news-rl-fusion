from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.train import evaluate_rl_fusion
from src.utils.config import load_yaml
from src.utils.file_io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained RL fusion checkpoint on an external split.")
    parser.add_argument("--config", default="configs/external_openfake_fusion.yaml")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    result = evaluate_rl_fusion(config, split=args.split)
    write_json(result, config["paths"]["metrics_path"])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
