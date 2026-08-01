from __future__ import annotations

import pytest
import torch

from src.text_model.dataset import TextNewsDataset

transformers = pytest.importorskip("transformers")
from src.text_model.model import TextClassifier


class DummyTokenizer:
    def __call__(self, text, padding, truncation, max_length, return_tensors):
        return {
            "input_ids": torch.ones((1, max_length), dtype=torch.long),
            "attention_mask": torch.ones((1, max_length), dtype=torch.long),
        }


class DummyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"hidden_size": 4})()

    def forward(self, input_ids, attention_mask):
        batch_size, sequence_length = input_ids.shape
        return type(
            "Output",
            (), {"last_hidden_state": torch.ones(batch_size, sequence_length, 4)},
        )()


def test_text_dataset_tokenizes_rows() -> None:
    rows = __import__("pandas").DataFrame(
        {"sample_id": ["a"], "text": ["hello world"], "label": [1]}
    )
    dataset = TextNewsDataset(rows, tokenizer=DummyTokenizer(), max_length=8)
    item = dataset[0]

    assert item["input_ids"].shape == (8,)
    assert item["attention_mask"].shape == (8,)
    assert item["label"].item() == 1


def test_text_classifier_forward_with_dummy_backbone(monkeypatch) -> None:
    def fake_from_pretrained(_architecture):
        return DummyBackbone()

    monkeypatch.setattr("src.text_model.model.AutoModel.from_pretrained", fake_from_pretrained)
    model = TextClassifier("dummy")
    outputs = model(torch.ones((2, 8), dtype=torch.long), torch.ones((2, 8), dtype=torch.long))

    assert set(outputs) == {"logits", "fake_probability", "predicted_label", "confidence", "embedding"}
    assert outputs["logits"].shape == (2, 2)
    assert torch.all((outputs["fake_probability"] >= 0) & (outputs["fake_probability"] <= 1))
