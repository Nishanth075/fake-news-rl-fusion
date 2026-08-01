from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


class TextNewsDataset(Dataset):
    """PyTorch dataset for DistilBERT text classification."""

    def __init__(self, rows: Any, tokenizer: Any, max_length: int) -> None:
        self.rows = rows.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows.iloc[index]
        encoded = self.tokenizer(
            str(row["text"]),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "sample_id": str(row["sample_id"]),
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }
