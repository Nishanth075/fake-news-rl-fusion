from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selected research pipeline stages.")
    parser.add_argument("--prepare-data", action="store_true", help="Prepare dataset CSV and splits.")
    parser.add_argument(
        "--prepare-fakeddit",
        action="store_true",
        help="Convert official Fakeddit TSV splits to project CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare_data:
        subprocess.run(
            [sys.executable, "scripts/prepare_dataset.py", "--config", "configs/data.yaml"],
            check=True,
        )
    elif args.prepare_fakeddit:
        subprocess.run(
            [sys.executable, "scripts/prepare_fakeddit.py", "--config", "configs/fakeddit.yaml"],
            check=True,
        )
    else:
        print(
            "No stage selected. Examples: python run_pipeline.py --prepare-data "
            "or python run_pipeline.py --prepare-fakeddit"
        )


if __name__ == "__main__":
    main()
