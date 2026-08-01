from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.prepare import prepare_dataset
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare paired image-text dataset splits.")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    stats = prepare_dataset(config)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
