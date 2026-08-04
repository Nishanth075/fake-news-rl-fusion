from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from src.utils.file_io import write_json
from src.utils.seed import set_seed


SPLITS = ["train", "validation", "test"]
EPSILON = 1e-8


def build_reliability_outputs(config: dict[str, Any]) -> dict[str, Any]:
    """Merge modality predictions with reliability features for fusion."""
    seed = int(config.get("seed", 42))
    set_seed(seed)
    rel_config = config["reliability"]
    splits = rel_config.get("splits", SPLITS)

    split_frames = {split: _load_split(rel_config, split) for split in splits}
    normalization_split = "train" if "train" in split_frames else splits[0]
    train_raw_image = _extract_image_raw_features(split_frames[normalization_split], rel_config)
    image_norm = _fit_image_normalizers(train_raw_image, rel_config)

    output_dir = Path(rel_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {"splits": {}, "image_normalization": image_norm}

    for split, split_df in split_frames.items():
        raw_image = train_raw_image if split == normalization_split else _extract_image_raw_features(split_df, rel_config)
        image_quality = _normalize_image_features(raw_image, image_norm, rel_config)
        text_quality = _extract_text_quality(split_df, rel_config)
        merged = _merge_outputs(split, split_df, image_quality, text_quality, rel_config)
        merged.to_csv(output_dir / f"{split}_outputs.csv", index=False)
        stats["splits"][split] = {
            "rows": int(len(merged)),
            "image_quality_mean": float(merged["image_quality"].mean()),
            "text_quality_mean": float(merged["text_quality"].mean()),
            "output_path": str(output_dir / f"{split}_outputs.csv"),
        }

    write_json(stats, rel_config["metrics_path"])
    return stats


def _load_split(rel_config: dict[str, Any], split: str) -> pd.DataFrame:
    splits_dir = Path(rel_config["splits_dir"])
    split_path = splits_dir / f"{split}.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_path}")
    return pd.read_csv(split_path)


def _extract_image_raw_features(df: pd.DataFrame, rel_config: dict[str, Any]) -> pd.DataFrame:
    image_root = Path(rel_config.get("image_root", "."))
    rows = []
    for row in df.itertuples(index=False):
        sample_id = str(getattr(row, "sample_id"))
        image_path = Path(str(getattr(row, "image_path")))
        full_path = image_path if image_path.is_absolute() else image_root / image_path
        features = _read_image_features(full_path)
        features["sample_id"] = sample_id
        rows.append(features)
    return pd.DataFrame(rows)


def _read_image_features(path: Path) -> dict[str, float]:
    if not path.exists():
        return {
            "image_available": 0.0,
            "blur_raw": 0.0,
            "brightness_raw": 0.0,
            "contrast_raw": 0.0,
            "entropy_raw": 0.0,
        }
    try:
        with Image.open(path) as image:
            gray = image.convert("L")
            array = np.asarray(gray, dtype=np.float32)
    except Exception:
        return {
            "image_available": 0.0,
            "blur_raw": 0.0,
            "brightness_raw": 0.0,
            "contrast_raw": 0.0,
            "entropy_raw": 0.0,
        }

    return {
        "image_available": 1.0,
        "blur_raw": _laplacian_variance(array),
        "brightness_raw": float(array.mean()),
        "contrast_raw": float(array.std()),
        "entropy_raw": _entropy(array),
    }


def _laplacian_variance(array: np.ndarray) -> float:
    padded = np.pad(array, 1, mode="edge")
    laplacian = (
        -4 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return float(laplacian.var())


def _entropy(array: np.ndarray) -> float:
    histogram, _ = np.histogram(array, bins=256, range=(0, 255), density=True)
    histogram = histogram[histogram > 0]
    return float(-(histogram * np.log2(histogram)).sum())


def _fit_image_normalizers(raw_df: pd.DataFrame, rel_config: dict[str, Any]) -> dict[str, dict[str, float]]:
    image_config = rel_config["image_quality"]
    available = raw_df[raw_df["image_available"] > 0]
    return {
        "blur": _percentile_bounds(available["blur_raw"], image_config["blur_percentiles"]),
        "contrast": _percentile_bounds(available["contrast_raw"], image_config["contrast_percentiles"]),
        "entropy": _percentile_bounds(available["entropy_raw"], image_config["entropy_percentiles"]),
    }


def _percentile_bounds(series: pd.Series, percentiles: list[int]) -> dict[str, float]:
    low, high = np.percentile(series.to_numpy(dtype=float), percentiles)
    if np.isclose(low, high):
        high = low + 1.0
    return {"low": float(low), "high": float(high)}


def _normalize_image_features(
    raw_df: pd.DataFrame,
    image_norm: dict[str, dict[str, float]],
    rel_config: dict[str, Any],
) -> pd.DataFrame:
    image_config = rel_config["image_quality"]
    result = raw_df.copy()
    result["image_blur_quality"] = _normalize_range(result["blur_raw"], image_norm["blur"])
    result["image_contrast_quality"] = _normalize_range(result["contrast_raw"], image_norm["contrast"])
    result["image_entropy_quality"] = _normalize_range(result["entropy_raw"], image_norm["entropy"])
    result["image_brightness_quality"] = result["brightness_raw"].apply(
        lambda value: _brightness_quality(
            value,
            low=float(image_config["brightness_low"]),
            mid=float(image_config["brightness_mid"]),
            high=float(image_config["brightness_high"]),
        )
    )
    quality_cols = [
        "image_blur_quality",
        "image_brightness_quality",
        "image_contrast_quality",
        "image_entropy_quality",
        "image_available",
    ]
    result["image_quality"] = result[quality_cols].mean(axis=1).clip(0, 1)
    return result[
        [
            "sample_id",
            "image_quality",
            "image_blur_quality",
            "image_brightness_quality",
            "image_contrast_quality",
            "image_entropy_quality",
            "image_available",
        ]
    ]


def _normalize_range(series: pd.Series, bounds: dict[str, float]) -> pd.Series:
    return ((series - bounds["low"]) / (bounds["high"] - bounds["low"] + EPSILON)).clip(0, 1)


def _brightness_quality(value: float, low: float, mid: float, high: float) -> float:
    if value <= low or value >= high:
        return 0.0
    if value <= mid:
        return float((value - low) / (mid - low + EPSILON))
    return float((high - value) / (high - mid + EPSILON))


def _extract_text_quality(df: pd.DataFrame, rel_config: dict[str, Any]) -> pd.DataFrame:
    target_words = float(rel_config.get("text_length_target_words", 30))
    max_length = float(rel_config.get("text_max_length", 128))
    rows = []
    for row in df.itertuples(index=False):
        text = str(getattr(row, "text"))
        words = text.split()
        word_count = len(words)
        token_count = word_count
        text_available = 1.0 if word_count > 0 else 0.0
        length_quality = min(word_count / target_words, 1.0) if text_available else 0.0
        truncation_quality = min(max_length / max(token_count, 1), 1.0) if text_available else 0.0
        text_quality = float(np.mean([length_quality, truncation_quality, text_available]))
        rows.append(
            {
                "sample_id": str(getattr(row, "sample_id")),
                "text_quality": text_quality,
                "text_word_count": word_count,
                "text_token_count": token_count,
                "text_length_quality": length_quality,
                "text_truncation_quality": truncation_quality,
                "text_available": text_available,
            }
        )
    return pd.DataFrame(rows)


def _merge_outputs(
    split: str,
    split_df: pd.DataFrame,
    image_quality: pd.DataFrame,
    text_quality: pd.DataFrame,
    rel_config: dict[str, Any],
) -> pd.DataFrame:
    image_outputs = pd.read_csv(Path(rel_config["image_outputs_dir"]) / f"{split}_image_outputs.csv")
    text_outputs = pd.read_csv(Path(rel_config["text_outputs_dir"]) / f"{split}_text_outputs.csv")
    base_cols = ["sample_id", "image_path", "text", "label"]
    base = split_df[base_cols].copy()
    merged = base.merge(image_outputs, on=["sample_id", "label"], how="inner")
    merged = merged.merge(text_outputs, on=["sample_id", "label"], how="inner")
    merged = merged.merge(image_quality, on="sample_id", how="inner")
    merged = merged.merge(text_quality, on="sample_id", how="inner")
    return merged
