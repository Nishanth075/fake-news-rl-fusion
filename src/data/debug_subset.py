from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.file_io import write_json
from src.utils.seed import set_seed


SPLIT_FILES = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv",
}


def create_debug_subset(config: dict[str, Any]) -> dict[str, Any]:
    """Create small reproducible split files for smoke testing in Colab."""
    seed = int(config.get("seed", 42))
    set_seed(seed)

    subset_config = config["subset"]
    input_dir = Path(subset_config["input_splits_dir"])
    output_dir = Path(subset_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    sizes = {
        "train": int(subset_config["train_size"]),
        "validation": int(subset_config["validation_size"]),
        "test": int(subset_config["test_size"]),
    }
    stratify = bool(subset_config.get("stratify", True))

    stats: dict[str, Any] = {"splits": {}}
    for split_name, file_name in SPLIT_FILES.items():
        source_path = input_dir / file_name
        if not source_path.exists():
            raise FileNotFoundError(f"Split CSV not found: {source_path}")
        df = pd.read_csv(source_path)
        subset_df = _sample_split(df, sizes[split_name], seed, stratify)
        subset_df.to_csv(output_dir / file_name, index=False)
        stats["splits"][split_name] = {
            "source_rows": int(len(df)),
            "subset_rows": int(len(subset_df)),
            "class_distribution": _label_counts(subset_df),
        }

    write_json(stats, output_dir / "debug_subset_stats.json")
    return stats


def _sample_split(df: pd.DataFrame, size: int, seed: int, stratify: bool) -> pd.DataFrame:
    if size <= 0 or len(df) <= size:
        return df.reset_index(drop=True)
    if not stratify:
        return df.sample(n=size, random_state=seed).reset_index(drop=True)

    labels = sorted(df["label"].unique())
    per_label = size // len(labels)
    remainder = size % len(labels)
    parts: list[pd.DataFrame] = []
    for index, label in enumerate(labels):
        label_df = df[df["label"] == label]
        label_size = per_label + (1 if index < remainder else 0)
        label_size = min(label_size, len(label_df))
        parts.append(label_df.sample(n=label_size, random_state=seed + int(label)))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _label_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df["label"].value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}
