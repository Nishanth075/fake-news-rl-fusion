from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.evaluation.metrics import binary_classification_metrics
from src.image_model.dataset import ImageNewsDataset
from src.image_model.model import build_image_model
from src.image_model.transforms import build_eval_transforms
from src.utils.device import get_device
from src.utils.file_io import write_json
from src.utils.seed import set_seed


SPLIT_PATH_KEYS = {
    "train": "train_csv",
    "validation": "validation_csv",
    "test": "test_csv",
}


def run_image_inference(config: dict[str, Any], splits: list[str] | None = None) -> dict[str, Any]:
    """Run image model inference and save probabilities for selected splits."""
    seed = int(config.get("seed", 42))
    set_seed(seed)

    paths = config["paths"]
    image_config = config["image_model"]
    requested_splits = splits or ["train", "validation", "test"]
    device = get_device()

    checkpoint_path = Path(paths["checkpoint_dir"]) / "best_image_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Image checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_image_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = Path(paths.get("outputs_dir", "data/modality_outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, Any] = {"device": str(device), "splits": {}}
    for split_name in requested_splits:
        if split_name not in SPLIT_PATH_KEYS:
            raise ValueError(f"Unknown split: {split_name}")
        split_csv = paths[SPLIT_PATH_KEYS[split_name]]
        split_df = pd.read_csv(split_csv)
        output_df = _predict_split(config, model, split_df, device)
        output_path = output_dir / f"{split_name}_image_outputs.csv"
        output_df.to_csv(output_path, index=False)

        split_metrics = binary_classification_metrics(
            output_df["label"].to_numpy(), output_df["image_probability"].to_numpy()
        )
        metrics["splits"][split_name] = {
            "rows": int(len(output_df)),
            "output_path": str(output_path),
            "metrics": split_metrics,
        }

    metrics_path = Path(paths.get("image_outputs_metrics_path", "outputs/metrics/image_outputs_metrics.json"))
    write_json(metrics, metrics_path)
    return metrics


def _predict_split(
    config: dict[str, Any],
    model: torch.nn.Module,
    split_df: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    image_config = config["image_model"]
    paths = config["paths"]
    transform = build_eval_transforms(int(image_config["resize_size"]), int(image_config["image_size"]))
    dataset = ImageNewsDataset(split_df, image_root=paths.get("image_root", "."), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=int(image_config["batch_size"]),
        shuffle=False,
        num_workers=int(image_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )

    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="image inference", leave=False):
            images = batch["image"].to(device)
            outputs = model(images)
            probabilities = outputs["fake_probability"].detach().cpu().numpy().tolist()
            predictions = outputs["predicted_label"].detach().cpu().numpy().astype(int).tolist()
            confidences = outputs["confidence"].detach().cpu().numpy().tolist()
            labels = batch["label"].detach().cpu().numpy().astype(int).tolist()
            sample_ids = batch["sample_id"]
            for sample_id, label, probability, prediction, confidence in zip(
                sample_ids, labels, probabilities, predictions, confidences
            ):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "label": label,
                        "image_probability": float(probability),
                        "image_prediction": int(prediction),
                        "image_confidence": float(confidence),
                    }
                )
    return pd.DataFrame(rows)
