from __future__ import annotations

import pandas as pd

from src.data.debug_subset import create_debug_subset
from src.data.download_images import download_split_images
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
    df = _example_split_df(20)
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


def test_download_images_writes_available_splits_for_existing_files(tmp_path) -> None:
    splits_dir = tmp_path / "splits"
    image_dir = tmp_path / "images"
    available_dir = tmp_path / "available"
    splits_dir.mkdir()
    image_dir.mkdir()
    df = _example_split_df(2)
    for image_name in ["id_0.jpg", "id_1.jpg"]:
        (image_dir / image_name).write_bytes(b"already present")
    for name in ["train", "validation", "test"]:
        df.to_csv(splits_dir / f"{name}.csv", index=False)

    stats = download_split_images(
        {
            "images": {
                "splits_dir": str(splits_dir),
                "output_dir": str(image_dir),
                "available_splits_dir": str(available_dir),
                "verify_images": False,
            }
        }
    )

    assert stats["total"]["available_rows"] == 6
    assert (available_dir / "train.csv").exists()
    assert len(pd.read_csv(available_dir / "train.csv")) == 2


def _example_split_df(size: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"id_{index}" for index in range(size)],
            "image_path": [f"data/images/fakeddit/id_{index}.jpg" for index in range(size)],
            "text": [f"text {index}" for index in range(size)],
            "label": [index % 2 for index in range(size)],
            "image_url": [f"https://example.com/{index}.jpg" for index in range(size)],
        }
    )
