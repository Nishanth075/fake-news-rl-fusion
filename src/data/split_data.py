from __future__ import annotations

import numpy as np
import pandas as pd


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create reproducible stratified train/validation/test splits."""
    total = train_ratio + validation_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    if "label" not in df.columns:
        raise ValueError("DataFrame must contain a label column")

    rng = np.random.default_rng(seed)
    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, group in df.groupby("label", sort=True):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        count = len(indices)
        train_end = int(round(count * train_ratio))
        validation_end = train_end + int(round(count * validation_ratio))

        if count >= 3:
            train_end = min(max(train_end, 1), count - 2)
            validation_end = min(max(validation_end, train_end + 1), count - 1)

        train_parts.append(df.loc[indices[:train_end]])
        validation_parts.append(df.loc[indices[train_end:validation_end]])
        test_parts.append(df.loc[indices[validation_end:]])

    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    validation_df = (
        pd.concat(validation_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    )
    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train_df, validation_df, test_df


def apply_debug_limits(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_size: int,
    validation_size: int,
    test_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Limit split sizes for quick Colab smoke tests."""
    return (
        _sample_split(train_df, train_size, seed),
        _sample_split(validation_df, validation_size, seed),
        _sample_split(test_df, test_size, seed),
    )


def _sample_split(df: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    if size <= 0 or len(df) <= size:
        return df.reset_index(drop=True)
    return df.sample(n=size, random_state=seed).reset_index(drop=True)
