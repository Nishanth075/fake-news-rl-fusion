from __future__ import annotations

import pandas as pd

from src.data.debug_subset import create_debug_subset
from src.data.validation import validate_paired_dataset


def test_validation_preserves_extra_columns() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["a"],
            "image_path": ["a.jpg"],
            "text": ["hello"],
            "label": [0],
            "image_url": ["https://example.com/a.jpg"],
        }
    )

    clean_df, _ = validate_paired_dataset(df)

    assert "image_url" in clean_df.columns


def test_create_debug_subset_balances_labels(tmp_path) -> None:
    input_dir = tmp_path / "splits"
    input_dir.mkdir()
    df = pd.DataFrame(
        {
            "sample_id": [f"id_{index}" for index in range(20)],
            "image_path": [f"data/images/fakeddit/id_{index}.jpg" for index in range(20)],
            "text": [f"text {index}" for index in range(20)],
            "label": [0] * 10 + [1] * 10,
            "image_url": [f"https://example.com/{index}.jpg" for index in range(20)],
        }
    )
    for name in ["train", "validation", "test"]:
        df.to_csv(input_dir / f"{name}.csv", index=False)

    config = {
        "seed": 42,
        "subset": {
            "input_splits_dir": str(input_dir),
            "output_dir": str(tmp_path / "debug_splits"),
            "train_size": 6,
            "validation_size": 4,
            "test_size": 4,
            "stratify": True,
        },
    }

    stats = create_debug_subset(config)
    train = pd.read_csv(tmp_path / "debug_splits" / "train.csv")

    assert stats["splits"]["train"]["subset_rows"] == 6
    assert train["label"].value_counts().to_dict() == {0: 3, 1: 3}
    assert "image_url" in train.columns
