from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.explainability.faithfulness import run_explainability_faithfulness
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deletion-style faithfulness tests for explanations.")
    parser.add_argument("--config", default="configs/final_explainability_faithfulness.yaml")
    return parser.parse_args()


def main() -> None:
    result = run_explainability_faithfulness(load_yaml(parse_args().config))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
