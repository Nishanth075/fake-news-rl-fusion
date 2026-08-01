from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageNewsDataset(Dataset):
    """PyTorch dataset for image-only fake-news classification."""

    def __init__(self, rows: Any, image_root: str | Path, transform: Any | None = None) -> None:
        self.rows = rows.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        image_path = Path(str(row["image_path"]))
        if not image_path.is_absolute():
            image_path = self.image_root / image_path

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        return {
            "sample_id": str(row["sample_id"]),
            "image": image,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
        }
