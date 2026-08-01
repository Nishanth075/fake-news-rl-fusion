from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selected research pipeline stages.")
    parser.add_argument("--prepare-data", action="store_true", help="Prepare dataset CSV and splits.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare_data:
        subprocess.run(
            [sys.executable, "scripts/prepare_dataset.py", "--config", "configs/data.yaml"],
            check=True,
        )
    else:
        print("No stage selected. Example: python run_pipeline.py --prepare-data")


if __name__ == "__main__":
    main()
