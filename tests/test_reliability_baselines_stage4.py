from __future__ import annotations

import pandas as pd
from PIL import Image

from src.baselines.methods import add_baseline_probabilities
from src.fusion.reliability import build_reliability_outputs


def test_baseline_probabilities_stay_in_range() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "label": [0, 1],
            "image_probability": [0.2, 0.8],
            "image_confidence": [0.8, 0.8],
            "image_quality": [0.5, 1.0],
            "text_probability": [0.4, 0.7],
            "text_confidence": [0.6, 0.7],
            "text_quality": [1.0, 0.5],
        }
    )

    output = add_baseline_probabilities(df)

    for column in output.columns:
        if column not in {"sample_id", "label"}:
            assert output[column].between(0, 1).all()


def test_reliability_builder_writes_merged_outputs(tmp_path) -> None:
    splits_dir = tmp_path / "splits"
    images_dir = tmp_path / "data" / "images" / "fakeddit"
    modality_dir = tmp_path / "modality"
    output_dir = tmp_path / "merged"
    splits_dir.mkdir()
    images_dir.mkdir(parents=True)
    modality_dir.mkdir()

    rows = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "image_path": ["data/images/fakeddit/a.jpg", "data/images/fakeddit/b.jpg"],
            "text": ["short text", "another short text"],
            "label": [0, 1],
        }
    )
    Image.new("RGB", (16, 16), color=(128, 128, 128)).save(images_dir / "a.jpg")
    Image.new("RGB", (16, 16), color=(255, 255, 255)).save(images_dir / "b.jpg")

    for split in ["train", "validation", "test"]:
        rows.to_csv(splits_dir / f"{split}.csv", index=False)
        pd.DataFrame(
            {
                "sample_id": ["a", "b"],
                "label": [0, 1],
                "image_probability": [0.2, 0.8],
                "image_prediction": [0, 1],
                "image_confidence": [0.8, 0.8],
            }
        ).to_csv(modality_dir / f"{split}_image_outputs.csv", index=False)
        pd.DataFrame(
            {
                "sample_id": ["a", "b"],
                "label": [0, 1],
                "text_probability": [0.3, 0.7],
                "text_prediction": [0, 1],
                "text_confidence": [0.7, 0.7],
            }
        ).to_csv(modality_dir / f"{split}_text_outputs.csv", index=False)

    stats = build_reliability_outputs(
        {
            "seed": 42,
            "reliability": {
                "splits_dir": str(splits_dir),
                "image_root": str(tmp_path),
                "image_outputs_dir": str(modality_dir),
                "text_outputs_dir": str(modality_dir),
                "output_dir": str(output_dir),
                "metrics_path": str(tmp_path / "metrics.json"),
                "text_length_target_words": 30,
                "text_max_length": 128,
                "image_quality": {
                    "blur_percentiles": [5, 95],
                    "contrast_percentiles": [5, 95],
                    "entropy_percentiles": [5, 95],
                    "brightness_low": 40,
                    "brightness_high": 215,
                    "brightness_mid": 127.5,
                },
            },
        }
    )
    merged = pd.read_csv(output_dir / "train_outputs.csv")

    assert stats["splits"]["train"]["rows"] == 2
    assert "image_quality" in merged.columns
    assert "text_quality" in merged.columns
    assert merged["image_quality"].between(0, 1).all()
    assert merged["text_quality"].between(0, 1).all()
