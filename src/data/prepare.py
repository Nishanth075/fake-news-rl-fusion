from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.preprocessing import clean_text, normalize_label
from src.data.split_data import apply_debug_limits, stratified_split
from src.data.validation import validate_paired_dataset
from src.utils.file_io import ensure_parent_dir, write_json
from src.utils.seed import set_seed


def prepare_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Prepare standardized dataset CSV and leakage-aware split files."""
    seed = int(config.get("seed", 42))
    set_seed(seed)

    data_config = config["data"]
    source_csv = Path(data_config["source_csv"])
    if not source_csv.exists():
        raise FileNotFoundError(
            f"Source CSV not found: {source_csv}. Put metadata there or update configs/data.yaml."
        )

    raw_df = pd.read_csv(source_csv)
    standardized = _standardize_columns(raw_df, data_config)

    validation_config = data_config.get("validation", {})
    clean_df, validation_report = validate_paired_dataset(
        standardized,
        image_root=data_config.get("image_root", "."),
        require_existing_images=bool(validation_config.get("require_existing_images", False)),
        min_text_chars=int(validation_config.get("min_text_chars", 1)),
        remove_duplicate_text=bool(validation_config.get("remove_duplicate_text", True)),
        remove_duplicate_image_paths=bool(
            validation_config.get("remove_duplicate_image_paths", True)
        ),
    )

    output_csv = Path(data_config["output_csv"])
    ensure_parent_dir(output_csv)
    clean_df.to_csv(output_csv, index=False)

    split_config = data_config["split"]
    train_df, validation_df, test_df = stratified_split(
        clean_df,
        train_ratio=float(split_config["train"]),
        validation_ratio=float(split_config["validation"]),
        test_ratio=float(split_config["test"]),
        seed=seed,
    )

    debug_config = config.get("debug", {})
    if bool(debug_config.get("enabled", False)):
        train_df, validation_df, test_df = apply_debug_limits(
            train_df,
            validation_df,
            test_df,
            train_size=int(debug_config.get("train_size", 500)),
            validation_size=int(debug_config.get("validation_size", 100)),
            test_size=int(debug_config.get("test_size", 100)),
            seed=seed,
        )

    splits_dir = Path(data_config["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(splits_dir / "train.csv", index=False)
    validation_df.to_csv(splits_dir / "validation.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    stats = _build_stats(clean_df, train_df, validation_df, test_df)
    stats["validation"] = validation_report.to_dict()
    stats_path = Path(data_config["stats_path"])
    write_json(stats, stats_path)
    return stats


def _standardize_columns(raw_df: pd.DataFrame, data_config: dict[str, Any]) -> pd.DataFrame:
    columns = data_config["columns"]
    missing = [source for source in columns.values() if source not in raw_df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing configured source columns: {missing}")

    label_config = data_config["label_mapping"]
    df = pd.DataFrame(
        {
            "sample_id": raw_df[columns["sample_id"]].astype(str).str.strip(),
            "image_path": raw_df[columns["image_path"]].astype(str).str.strip(),
            "text": raw_df[columns["text"]].apply(clean_text),
            "label": raw_df[columns["label"]].apply(
                lambda value: normalize_label(
                    value,
                    real_values=label_config["real_values"],
                    fake_values=label_config["fake_values"],
                )
            ),
        }
    )
    invalid_labels = int(df["label"].isna().sum())
    if invalid_labels:
        raise ValueError(
            f"Found {invalid_labels} labels that could not be mapped to 0=Real or 1=Fake."
        )
    df["label"] = df["label"].astype(int)
    return df


def _build_stats(
    clean_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "total_rows": int(len(clean_df)),
        "class_distribution": _label_counts(clean_df),
        "splits": {
            "train": _split_stats(train_df),
            "validation": _split_stats(validation_df),
            "test": _split_stats(test_df),
        },
    }


def _split_stats(df: pd.DataFrame) -> dict[str, Any]:
    return {"rows": int(len(df)), "class_distribution": _label_counts(df)}


def _label_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df["label"].value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}
