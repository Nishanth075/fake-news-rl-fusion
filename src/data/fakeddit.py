from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.preprocessing import clean_text
from src.data.validation import validate_paired_dataset
from src.utils.file_io import ensure_parent_dir, write_json
from src.utils.seed import set_seed


REQUIRED_FAKEDDIT_COLUMNS = ["id", "clean_title", "image_url", "2_way_label"]


def prepare_fakeddit(config: dict[str, Any]) -> dict[str, Any]:
    """Convert official Fakeddit multimodal TSV splits into project CSV files."""
    seed = int(config.get("seed", 42))
    set_seed(seed)

    fakeddit_config = config["fakeddit"]
    raw_dir = Path(fakeddit_config["raw_dir"])
    splits = {
        "train": raw_dir / fakeddit_config["train_tsv"],
        "validation": raw_dir / fakeddit_config["validation_tsv"],
        "test": raw_dir / fakeddit_config["test_tsv"],
    }

    validation_config = fakeddit_config.get("validation", {})
    prepared_splits: dict[str, pd.DataFrame] = {}
    validation_reports: dict[str, dict[str, int]] = {}

    for split_name, split_path in splits.items():
        raw_df = _read_fakeddit_tsv(split_path)
        standardized = _standardize_fakeddit_split(raw_df, fakeddit_config)
        clean_df, report = validate_paired_dataset(
            standardized,
            require_existing_images=False,
            min_text_chars=int(validation_config.get("min_text_chars", 1)),
            remove_duplicate_text=bool(validation_config.get("remove_duplicate_text", False)),
            remove_duplicate_image_paths=bool(
                validation_config.get("remove_duplicate_image_paths", False)
            ),
        )
        prepared_splits[split_name] = clean_df
        validation_reports[split_name] = report.to_dict()

    leakage_report = _cross_split_leakage_report(prepared_splits)
    combined = pd.concat(
        [df.assign(split=split_name) for split_name, df in prepared_splits.items()],
        ignore_index=True,
    )

    output_csv = Path(fakeddit_config["output_csv"])
    ensure_parent_dir(output_csv)
    combined.to_csv(output_csv, index=False)

    splits_dir = Path(fakeddit_config["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)
    prepared_splits["train"].to_csv(splits_dir / "train.csv", index=False)
    prepared_splits["validation"].to_csv(splits_dir / "validation.csv", index=False)
    prepared_splits["test"].to_csv(splits_dir / "test.csv", index=False)

    stats = {
        "dataset": "Fakeddit multimodal_only_samples",
        "label_standard": {"0": "Real", "1": "Fake"},
        "fakeddit_label_note": "Fakeddit 2_way_label is converted from 0=False/Fake, 1=True/Real to project standard 0=Real, 1=Fake.",
        "total_rows": int(len(combined)),
        "class_distribution": _label_counts(combined),
        "splits": {
            split_name: {
                "rows": int(len(split_df)),
                "class_distribution": _label_counts(split_df),
            }
            for split_name, split_df in prepared_splits.items()
        },
        "validation": validation_reports,
        "cross_split_leakage": leakage_report,
    }
    write_json(stats, fakeddit_config["stats_path"])
    return stats


def _read_fakeddit_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fakeddit TSV not found: {path}")
    df = pd.read_csv(path, sep="\t")
    missing = [column for column in REQUIRED_FAKEDDIT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required Fakeddit columns: {missing}")
    return df


def _standardize_fakeddit_split(
    raw_df: pd.DataFrame,
    fakeddit_config: dict[str, Any],
) -> pd.DataFrame:
    image_output_dir = str(fakeddit_config["image_output_dir"]).strip().replace("\\", "/")
    image_extension = str(fakeddit_config.get("image_extension", ".jpg"))
    invert_label = bool(fakeddit_config.get("invert_2_way_label", True))

    sample_ids = raw_df["id"].astype(str).str.strip()
    labels = raw_df["2_way_label"].astype(int)
    if invert_label:
        labels = 1 - labels

    standardized = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "image_path": sample_ids.apply(
                lambda sample_id: f"{image_output_dir}/{sample_id}{image_extension}"
            ),
            "text": raw_df["clean_title"].apply(clean_text),
            "label": labels.astype(int),
            "image_url": raw_df["image_url"].astype(str).str.strip(),
            "fakeddit_2_way_label": raw_df["2_way_label"].astype(int),
        }
    )
    if "hasImage" in raw_df.columns:
        standardized["hasImage"] = raw_df["hasImage"]
    return standardized


def _cross_split_leakage_report(splits: dict[str, pd.DataFrame]) -> dict[str, int]:
    reports: dict[str, int] = {}
    split_names = list(splits)
    for left_index, left_name in enumerate(split_names):
        for right_name in split_names[left_index + 1 :]:
            left = splits[left_name]
            right = splits[right_name]
            prefix = f"{left_name}_vs_{right_name}"
            reports[f"{prefix}_sample_id_overlap"] = int(
                len(set(left["sample_id"]) & set(right["sample_id"]))
            )
            reports[f"{prefix}_image_path_overlap"] = int(
                len(set(left["image_path"]) & set(right["image_path"]))
            )
            reports[f"{prefix}_text_overlap"] = int(
                len(set(left["text"].str.lower()) & set(right["text"].str.lower()))
            )
    return reports


def _label_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df["label"].value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}
