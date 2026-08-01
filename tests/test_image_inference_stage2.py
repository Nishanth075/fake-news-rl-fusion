from __future__ import annotations

import pandas as pd
import torch
from PIL import Image

from src.image_model.infer import run_image_inference
from src.image_model.model import ImageClassifier


def test_image_inference_writes_outputs(tmp_path) -> None:
    image_dir = tmp_path / "data" / "images" / "fakeddit"
    split_dir = tmp_path / "splits"
    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "outputs"
    image_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)

    Image.new("RGB", (32, 32), color=(0, 255, 0)).save(image_dir / "sample.jpg")
    split_df = pd.DataFrame(
        {
            "sample_id": ["sample"],
            "image_path": ["data/images/fakeddit/sample.jpg"],
            "text": ["sample text"],
            "label": [1],
        }
    )
    for split_name in ["train", "validation", "test"]:
        split_df.to_csv(split_dir / f"{split_name}.csv", index=False)

    model = ImageClassifier(pretrained=False)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": {}, "epoch": 1},
        checkpoint_dir / "best_image_model.pt",
    )

    config = {
        "seed": 42,
        "image_model": {
            "architecture": "resnet18",
            "pretrained": False,
            "image_size": 32,
            "resize_size": 32,
            "batch_size": 1,
            "num_workers": 0,
        },
        "paths": {
            "train_csv": str(split_dir / "train.csv"),
            "validation_csv": str(split_dir / "validation.csv"),
            "test_csv": str(split_dir / "test.csv"),
            "image_root": str(tmp_path),
            "checkpoint_dir": str(checkpoint_dir),
            "outputs_dir": str(output_dir),
            "image_outputs_metrics_path": str(tmp_path / "metrics.json"),
        },
    }

    metrics = run_image_inference(config, splits=["test"])
    outputs = pd.read_csv(output_dir / "test_image_outputs.csv")

    assert metrics["splits"]["test"]["rows"] == 1
    assert set(outputs.columns) == {
        "sample_id",
        "label",
        "image_probability",
        "image_prediction",
        "image_confidence",
    }
