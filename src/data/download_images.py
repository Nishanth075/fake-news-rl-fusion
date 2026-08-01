from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from PIL import Image
from requests import Response

from src.utils.file_io import write_json


SPLIT_FILES = ["train.csv", "validation.csv", "test.csv"]


def download_split_images(config: dict[str, Any]) -> dict[str, Any]:
    """Download images referenced by split CSV files that contain image_url."""
    image_config = config["images"]
    splits_dir = Path(image_config["splits_dir"])
    output_dir = Path(image_config["output_dir"])
    available_splits_dir = Path(image_config.get("available_splits_dir", splits_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    available_splits_dir.mkdir(parents=True, exist_ok=True)

    timeout = int(image_config.get("timeout_seconds", 15))
    max_retries = int(image_config.get("max_retries", 2))
    verify_images = bool(image_config.get("verify_images", True))
    headers = {"User-Agent": str(image_config.get("user_agent", "fake-news-rl-fusion"))}

    stats: dict[str, Any] = {
        "splits": {},
        "total": {"requested": 0, "downloaded": 0, "failed": 0, "available_rows": 0},
    }
    for split_file in SPLIT_FILES:
        split_path = splits_dir / split_file
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        df = pd.read_csv(split_path)
        if "image_url" not in df.columns:
            raise ValueError(f"{split_path} does not contain image_url. Re-run prepare_fakeddit first.")

        downloaded = 0
        available_mask = []
        failed_rows = []
        for row in df.itertuples(index=False):
            sample_id = str(getattr(row, "sample_id"))
            image_url = str(getattr(row, "image_url"))
            image_path = Path(str(getattr(row, "image_path")))
            target_path = output_dir / image_path.name

            ok, reason = _download_one(image_url, target_path, headers, timeout, max_retries)
            if ok and verify_images:
                ok, reason = _verify_image(target_path)
            if ok:
                downloaded += 1
                available_mask.append(True)
            else:
                available_mask.append(False)
                failed_rows.append({"sample_id": sample_id, "image_url": image_url, "reason": reason})

        available_df = df.loc[available_mask].copy().reset_index(drop=True)
        available_df.to_csv(available_splits_dir / split_file, index=False)

        requested = int(len(df))
        failed = len(failed_rows)
        available_rows = int(len(available_df))
        stats["splits"][split_file.replace(".csv", "")] = {
            "requested": requested,
            "downloaded": downloaded,
            "failed": failed,
            "available_rows": available_rows,
            "class_distribution": _label_counts(available_df),
        }
        stats["total"]["requested"] += requested
        stats["total"]["downloaded"] += downloaded
        stats["total"]["failed"] += failed
        stats["total"]["available_rows"] += available_rows
        if failed_rows:
            pd.DataFrame(failed_rows).to_csv(splits_dir / f"{split_file}_image_failures.csv", index=False)

    write_json(stats, available_splits_dir / "image_download_stats.json")
    return stats


def _download_one(
    image_url: str,
    target_path: Path,
    headers: dict[str, str],
    timeout: int,
    max_retries: int,
) -> tuple[bool, str]:
    if target_path.exists() and target_path.stat().st_size > 0:
        return True, "already_exists"
    if not image_url.startswith("http"):
        return False, "invalid_url"

    last_error = "unknown_error"
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(image_url, headers=headers, timeout=timeout, stream=True)
            ok, reason = _save_response(response, target_path)
            if ok:
                return True, "downloaded"
            last_error = reason
        except requests.RequestException as exc:
            last_error = exc.__class__.__name__
        if attempt < max_retries:
            time.sleep(1.0)
    return False, last_error


def _save_response(response: Response, target_path: Path) -> tuple[bool, str]:
    if response.status_code != 200:
        return False, f"http_{response.status_code}"
    content_type = response.headers.get("Content-Type", "")
    if "image" not in content_type.lower():
        return False, f"not_image_{content_type}"

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 64):
            if chunk:
                handle.write(chunk)
    return True, "saved"


def _verify_image(path: Path) -> tuple[bool, str]:
    try:
        with Image.open(path) as image:
            image.verify()
        return True, "verified"
    except Exception as exc:  # Pillow raises several image-specific exceptions.
        path.unlink(missing_ok=True)
        return False, exc.__class__.__name__


def _label_counts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {}
    counts = df["label"].value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}
