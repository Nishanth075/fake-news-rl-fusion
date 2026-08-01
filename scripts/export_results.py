from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting.export_results import export_debug_results
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export debug-stage research result tables.")
    parser.add_argument("--config", default="configs/results_export.yaml")
    return parser.parse_args()


def main() -> None:
    result = export_debug_results(load_yaml(parse_args().config))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
