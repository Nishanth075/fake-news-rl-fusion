from __future__ import annotations

import pandas as pd

from src.data.prepare import prepare_dataset
from src.data.split_data import stratified_split
from src.data.validation import validate_paired_dataset


def test_validation_removes_invalid_and_duplicates() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["a", "a", "b", "c", "d"],
            "image_path": ["1.jpg", "1-dup.jpg", "2.jpg", "3.jpg", ""],
            "text": ["Hello", "Duplicate id", "", "Hello", "Missing image"],
            "label": [0, 1, 0, 1, 0],
        }
    )

    clean_df, report = validate_paired_dataset(df)

    assert len(clean_df) == 1
    assert report.removed_duplicate_ids == 1
    assert report.removed_empty_text == 1
    assert report.removed_missing_required == 1


def test_stratified_split_is_reproducible() -> None:
    df = pd.DataFrame(
        {
            "sample_id": [f"id_{index}" for index in range(20)],
            "image_path": [f"{index}.jpg" for index in range(20)],
            "text": [f"text {index}" for index in range(20)],
            "label": [0] * 10 + [1] * 10,
        }
    )

    first = stratified_split(df, 0.7, 0.15, 0.15, seed=42)
    second = stratified_split(df, 0.7, 0.15, 0.15, seed=42)

    assert [split["sample_id"].tolist() for split in first] == [
        split["sample_id"].tolist() for split in second
    ]
    assert sum(len(split) for split in first) == len(df)


def test_prepare_dataset_writes_outputs(tmp_path) -> None:
    source_csv = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "id": [f"id_{index}" for index in range(12)],
            "image": [f"data/images/{index}.jpg" for index in range(12)],
            "title": [f"sample text {index}" for index in range(12)],
            "binary": [0, 1] * 6,
        }
    ).to_csv(source_csv, index=False)

    config = {
        "seed": 42,
        "data": {
            "source_csv": str(source_csv),
            "output_csv": str(tmp_path / "processed" / "dataset.csv"),
            "image_root": ".",
            "splits_dir": str(tmp_path / "splits"),
            "stats_path": str(tmp_path / "metrics" / "dataset_stats.json"),
            "columns": {
                "sample_id": "id",
                "image_path": "image",
                "text": "title",
                "label": "binary",
            },
            "label_mapping": {
                "real_values": [0, "0"],
                "fake_values": [1, "1"],
            },
            "split": {"train": 0.7, "validation": 0.15, "test": 0.15},
            "validation": {
                "require_existing_images": False,
                "min_text_chars": 1,
                "remove_duplicate_text": True,
                "remove_duplicate_image_paths": True,
            },
        },
        "debug": {"enabled": False},
    }

    stats = prepare_dataset(config)

    assert stats["total_rows"] == 12
    assert (tmp_path / "processed" / "dataset.csv").exists()
    assert (tmp_path / "splits" / "train.csv").exists()
    assert (tmp_path / "splits" / "validation.csv").exists()
    assert (tmp_path / "splits" / "test.csv").exists()
    assert (tmp_path / "metrics" / "dataset_stats.json").exists()
