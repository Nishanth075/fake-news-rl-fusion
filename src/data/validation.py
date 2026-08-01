from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["sample_id", "image_path", "text", "label"]


@dataclass(frozen=True)
class ValidationReport:
    input_rows: int
    valid_rows: int
    removed_rows: int
    removed_missing_required: int
    removed_empty_text: int
    removed_missing_images: int
    removed_duplicate_ids: int
    removed_duplicate_text: int
    removed_duplicate_image_paths: int

    def to_dict(self) -> dict[str, int]:
        return {
            "input_rows": self.input_rows,
            "valid_rows": self.valid_rows,
            "removed_rows": self.removed_rows,
            "removed_missing_required": self.removed_missing_required,
            "removed_empty_text": self.removed_empty_text,
            "removed_missing_images": self.removed_missing_images,
            "removed_duplicate_ids": self.removed_duplicate_ids,
            "removed_duplicate_text": self.removed_duplicate_text,
            "removed_duplicate_image_paths": self.removed_duplicate_image_paths,
        }


def _resolve_image_path(path_value: str, image_root: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(image_root) / path


def validate_paired_dataset(
    df: pd.DataFrame,
    image_root: str | Path = ".",
    require_existing_images: bool = False,
    min_text_chars: int = 1,
    remove_duplicate_text: bool = True,
    remove_duplicate_image_paths: bool = True,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate the standardized paired image-text dataset."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    original_rows = len(df)
    work = df.copy()
    work["sample_id"] = work["sample_id"].astype(str).str.strip()
    work["image_path"] = work["image_path"].astype(str).str.strip()
    work["text"] = work["text"].astype(str).str.strip()

    required_mask = work["sample_id"].ne("") & work["image_path"].ne("") & work["label"].notna()
    removed_missing_required = int((~required_mask).sum())
    work = work.loc[required_mask].copy()

    text_mask = work["text"].str.len() >= min_text_chars
    removed_empty_text = int((~text_mask).sum())
    work = work.loc[text_mask].copy()

    removed_missing_images = 0
    if require_existing_images:
        image_mask = work["image_path"].apply(
            lambda value: _resolve_image_path(value, image_root).is_file()
        )
        removed_missing_images = int((~image_mask).sum())
        work = work.loc[image_mask].copy()

    removed_duplicate_ids = int(work.duplicated("sample_id", keep="first").sum())
    work = work.drop_duplicates("sample_id", keep="first")

    removed_duplicate_text = 0
    if remove_duplicate_text:
        text_key = work["text"].str.lower()
        duplicate_text_mask = text_key.duplicated(keep="first")
        removed_duplicate_text = int(duplicate_text_mask.sum())
        work = work.loc[~duplicate_text_mask].copy()

    removed_duplicate_image_paths = 0
    if remove_duplicate_image_paths:
        image_key = work["image_path"].str.lower()
        duplicate_image_mask = image_key.duplicated(keep="first")
        removed_duplicate_image_paths = int(duplicate_image_mask.sum())
        work = work.loc[~duplicate_image_mask].copy()

    work = work[REQUIRED_COLUMNS].reset_index(drop=True)
    removed_rows = original_rows - len(work)
    report = ValidationReport(
        input_rows=original_rows,
        valid_rows=len(work),
        removed_rows=removed_rows,
        removed_missing_required=removed_missing_required,
        removed_empty_text=removed_empty_text,
        removed_missing_images=removed_missing_images,
        removed_duplicate_ids=removed_duplicate_ids,
        removed_duplicate_text=removed_duplicate_text,
        removed_duplicate_image_paths=removed_duplicate_image_paths,
    )
    return work, report

