from __future__ import annotations

import pandas as pd
import torch
from PIL import Image

from src.explainability.explain import compute_gradcam, overlay_heatmap, run_explainability
from src.image_model.model import ImageClassifier


class DummyTokenizer:
    def __call__(self, text, padding, truncation, max_length, return_tensors):
        return {
            "input_ids": torch.arange(max_length).unsqueeze(0),
            "attention_mask": torch.ones((1, max_length), dtype=torch.long),
        }

    def convert_ids_to_tokens(self, token_ids):
        return [f"tok{token_id}" for token_id in token_ids]


class DummyEmbeddings(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(32, 4)


class DummyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = DummyEmbeddings()
        self.config = type("Config", (), {"hidden_size": 4})()

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.embeddings.word_embeddings(input_ids)
        return type("Output", (), {"last_hidden_state": inputs_embeds})()


class DummyTextModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = DummyBackbone()
        self.classifier = torch.nn.Linear(4, 2)


def test_compute_gradcam_normalizes_map() -> None:
    activations = torch.ones((1, 2, 4, 4))
    gradients = torch.ones((1, 2, 4, 4))

    cam = compute_gradcam(activations, gradients)

    assert cam.shape == (4, 4)
    assert torch.all(cam == 0)


def test_overlay_heatmap_preserves_image_size() -> None:
    image = Image.new("RGB", (20, 10), color=(255, 255, 255))
    cam = torch.linspace(0, 1, steps=16).reshape(4, 4)

    overlay = overlay_heatmap(image, cam)

    assert overlay.size == image.size


def test_run_explainability_writes_summary(tmp_path, monkeypatch) -> None:
    image_dir = tmp_path / "images"
    split_dir = tmp_path / "splits"
    checkpoint_image_dir = tmp_path / "image_checkpoint"
    checkpoint_text_dir = tmp_path / "text_checkpoint"
    output_dir = tmp_path / "explain"
    image_dir.mkdir()
    split_dir.mkdir()
    checkpoint_image_dir.mkdir()
    checkpoint_text_dir.mkdir()

    Image.new("RGB", (32, 32), color=(80, 120, 160)).save(image_dir / "sample.jpg")
    pd.DataFrame(
        {
            "sample_id": ["sample"],
            "image_path": [str(image_dir / "sample.jpg")],
            "text": ["sample fake news text"],
            "label": [1],
        }
    ).to_csv(split_dir / "test.csv", index=False)

    image_model = ImageClassifier(pretrained=False)
    torch.save({"model_state_dict": image_model.state_dict()}, checkpoint_image_dir / "best_image_model.pt")
    text_model = DummyTextModel()
    torch.save({"model_state_dict": text_model.state_dict()}, checkpoint_text_dir / "best_text_model.pt")

    monkeypatch.setattr("src.explainability.explain._build_text_model", lambda _config: DummyTextModel())
    monkeypatch.setattr("src.explainability.explain._build_tokenizer", lambda _config: DummyTokenizer())

    config = {
        "seed": 42,
        "image_model": {
            "architecture": "resnet18",
            "pretrained": False,
            "image_size": 32,
            "resize_size": 32,
        },
        "text_model": {"architecture": "dummy", "max_length": 8},
        "paths": {
            "image_root": ".",
            "image_checkpoint_dir": str(checkpoint_image_dir),
            "text_checkpoint_dir": str(checkpoint_text_dir),
        },
        "explainability": {
            "sample_count": 1,
            "split_csv": str(split_dir / "test.csv"),
            "output_dir": str(output_dir),
            "summary_path": str(tmp_path / "summary.json"),
        },
    }

    summary = run_explainability(config)

    assert summary["rows"] == 1
    assert (output_dir / "explanation_summary.csv").exists()
    assert (output_dir / "image_gradcam" / "sample_gradcam.png").exists()
    assert (output_dir / "text_saliency" / "sample_text_saliency.csv").exists()

