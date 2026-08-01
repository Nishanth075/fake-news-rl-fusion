from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision import models


class ImageClassifier(nn.Module):
    """Binary ResNet image classifier that can expose embeddings."""

    def __init__(self, architecture: str = "resnet18", pretrained: bool = True) -> None:
        super().__init__()
        if architecture != "resnet18":
            raise ValueError(f"Unsupported image architecture: {architecture}")

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(in_features, 1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        embeddings = self.backbone(images)
        logits = self.classifier(embeddings).squeeze(1)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).long()
        confidence = torch.maximum(probabilities, 1.0 - probabilities)
        return {
            "logits": logits,
            "fake_probability": probabilities,
            "predicted_label": predictions,
            "confidence": confidence,
            "embedding": embeddings,
        }


def build_image_model(config: dict[str, Any]) -> ImageClassifier:
    image_config = config["image_model"]
    return ImageClassifier(
        architecture=str(image_config.get("architecture", "resnet18")),
        pretrained=bool(image_config.get("pretrained", True)),
    )
