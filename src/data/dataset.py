from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class PairedNewsDataset:
    """Lightweight CSV-backed paired image-text dataset."""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {self.csv_path}")
        self.data = pd.read_csv(self.csv_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.data.iloc[index]
        return {
            "sample_id": row["sample_id"],
            "image_path": row["image_path"],
            "text": row["text"],
            "label": int(row["label"]),
        }
