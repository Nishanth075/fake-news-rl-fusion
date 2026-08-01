from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.text_model.train import train_text_model
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DistilBERT text branch.")
    parser.add_argument("--config", default="configs/text_model.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Run one train/eval batch only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_text_model(load_yaml(args.config), smoke_test=args.smoke_test)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
