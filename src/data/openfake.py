from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from src.utils.file_io import write_json
from src.utils.seed import set_seed


def prepare_openfake_external(config: dict[str, Any]) -> dict[str, Any]:
    """Export a balanced OpenFake split into the project CSV/image format."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised in Colab/runtime use
        raise ImportError("Install Hugging Face datasets first: pip install datasets") from exc

    dataset_config = config["dataset"]
    paths = config["paths"]
    seed = int(config.get("seed", 42))
    set_seed(seed)

    output_dir = Path(paths["output_dir"])
    image_dir = Path(paths["image_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    requested_rows = int(dataset_config.get("sample_size", 3780))
    balance_labels = bool(dataset_config.get("balance_labels", True))
    text_column = str(dataset_config.get("text_column", "prompt"))
    fallback_text_column = str(dataset_config.get("fallback_text_column", "type"))

    ds = load_dataset(
        str(dataset_config.get("name", "ComplexDataLab/OpenFake")),
        str(dataset_config.get("config_name", "reddit")),
        split=str(dataset_config.get("split", "test")),
    )
    ds = ds.shuffle(seed=seed)

    target_per_label = requested_rows // 2 if balance_labels else requested_rows
    counts = {0: 0, 1: 0}
    rows: list[dict[str, Any]] = []

    for item in ds:
        label = _label_to_int(item.get("label"))
        if balance_labels and counts[label] >= target_per_label:
            if all(value >= target_per_label for value in counts.values()):
                break
            continue
        if not balance_labels and len(rows) >= requested_rows:
            break

        image = item.get("image")
        if image is None:
            continue
        image = _to_rgb_image(image)

        sample_id = f"openfake_{len(rows):05d}"
        image_path = image_dir / f"{sample_id}.jpg"
        image.save(image_path, quality=90)

        text = item.get(text_column) or item.get(fallback_text_column) or ""
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": image_path.as_posix(),
                "text": str(text),
                "label": label,
                "external_source": "ComplexDataLab/OpenFake",
                "external_split": str(dataset_config.get("split", "test")),
                "external_original_label": str(item.get("label")),
                "external_model": str(item.get("model", "")),
                "external_type": str(item.get("type", "")),
            }
        )
        counts[label] += 1

    if not rows:
        raise ValueError("No OpenFake rows were exported. Check dataset access and config.")

    output_csv = output_dir / "test.csv"
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    stats = {
        "dataset": str(dataset_config.get("name", "ComplexDataLab/OpenFake")),
        "config_name": str(dataset_config.get("config_name", "reddit")),
        "split": str(dataset_config.get("split", "test")),
        "requested_rows": requested_rows,
        "exported_rows": int(len(df)),
        "class_distribution": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "output_csv": str(output_csv),
        "image_dir": str(image_dir),
        "note": "External test-only dataset; not used for training or model selection.",
    }
    write_json(stats, paths["stats_path"])
    return stats


def _label_to_int(label: Any) -> int:
    text = str(label).strip().lower()
    if text in {"real", "0"}:
        return 0
    if text in {"fake", "1"}:
        return 1
    raise ValueError(f"Unsupported OpenFake label: {label!r}")


def _to_rgb_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")
