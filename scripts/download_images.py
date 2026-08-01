from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download_images import download_split_images
from src.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download images referenced by split CSV files.")
    parser.add_argument("--config", default="configs/image_download.yaml")
    return parser.parse_args()


def main() -> None:
    stats = download_split_images(load_yaml(parse_args().config))
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
