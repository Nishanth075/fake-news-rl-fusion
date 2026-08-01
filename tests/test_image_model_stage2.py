from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from src.evaluation.metrics import binary_classification_metrics
from src.image_model.dataset import ImageNewsDataset
from src.image_model.model import ImageClassifier
from src.image_model.transforms import build_eval_transforms


def test_image_model_forward_outputs_expected_keys() -> None:
    model = ImageClassifier(pretrained=False)
    model.eval()
    with torch.no_grad():
        outputs = model(torch.randn(2, 3, 224, 224))

    assert set(outputs) == {"logits", "fake_probability", "predicted_label", "confidence", "embedding"}
    assert outputs["logits"].shape == (2,)
    assert outputs["embedding"].shape[0] == 2
    assert torch.all((outputs["fake_probability"] >= 0) & (outputs["fake_probability"] <= 1))


def test_image_dataset_loads_rgb_image(tmp_path) -> None:
    image_dir = tmp_path / "data" / "images" / "fakeddit"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "sample.jpg"
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(image_path)

    rows = __import__("pandas").DataFrame(
        {
            "sample_id": ["sample"],
            "image_path": ["data/images/fakeddit/sample.jpg"],
            "text": ["hello"],
            "label": [1],
        }
    )
    dataset = ImageNewsDataset(rows, image_root=tmp_path, transform=build_eval_transforms(32, 32))
    item = dataset[0]

    assert item["image"].shape == (3, 32, 32)
    assert item["label"].item() == 1.0


def test_binary_metrics_include_macro_f1_and_confusion_matrix() -> None:
    metrics = binary_classification_metrics(np.array([0, 1, 1]), np.array([0.1, 0.8, 0.4]))

    assert "macro_f1" in metrics
    assert metrics["confusion_matrix"] == [[1, 0], [1, 1]]
