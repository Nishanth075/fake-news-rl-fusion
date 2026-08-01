from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class TextClassifier(nn.Module):
    """DistilBERT-based binary text classifier."""

    def __init__(self, architecture: str = "distilbert-base-uncased") -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(architecture)
        hidden_size = int(self.backbone.config.hidden_size)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        probabilities = torch.softmax(logits, dim=1)
        fake_probability = probabilities[:, 1]
        confidence, predicted_label = torch.max(probabilities, dim=1)
        return {
            "logits": logits,
            "fake_probability": fake_probability,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "embedding": cls_embedding,
        }


def build_text_model(config: dict[str, Any]) -> TextClassifier:
    return TextClassifier(str(config["text_model"].get("architecture", "distilbert-base-uncased")))


def build_tokenizer(config: dict[str, Any]):
    return AutoTokenizer.from_pretrained(str(config["text_model"].get("architecture", "distilbert-base-uncased")))
