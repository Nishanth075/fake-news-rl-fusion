from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.evaluation.metrics import binary_classification_metrics
from src.text_model.dataset import TextNewsDataset
from src.text_model.model import build_text_model, build_tokenizer
from src.utils.device import get_device
from src.utils.file_io import write_json
from src.utils.seed import set_seed


SPLIT_PATH_KEYS = {"train": "train_csv", "validation": "validation_csv", "test": "test_csv"}


def run_text_inference(config: dict[str, Any], splits: list[str] | None = None) -> dict[str, Any]:
    """Run text model inference and save probabilities for selected splits."""
    seed = int(config.get("seed", 42))
    set_seed(seed)

    paths = config["paths"]
    requested_splits = splits or ["train", "validation", "test"]
    device = get_device()
    tokenizer = build_tokenizer(config)

    checkpoint_path = Path(paths["checkpoint_dir"]) / "best_text_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Text checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_text_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = Path(paths.get("outputs_dir", "data/modality_outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, Any] = {"device": str(device), "splits": {}}
    for split_name in requested_splits:
        if split_name not in SPLIT_PATH_KEYS:
            raise ValueError(f"Unknown split: {split_name}")
        split_df = pd.read_csv(paths[SPLIT_PATH_KEYS[split_name]])
        output_df = _predict_split(config, tokenizer, model, split_df, device)
        output_path = output_dir / f"{split_name}_text_outputs.csv"
        output_df.to_csv(output_path, index=False)
        split_metrics = binary_classification_metrics(
            output_df["label"].to_numpy(), output_df["text_probability"].to_numpy()
        )
        metrics["splits"][split_name] = {
            "rows": int(len(output_df)),
            "output_path": str(output_path),
            "metrics": split_metrics,
        }

    write_json(metrics, paths.get("text_outputs_metrics_path", "outputs/metrics/text_outputs_metrics.json"))
    return metrics


def _predict_split(
    config: dict[str, Any],
    tokenizer: Any,
    model: torch.nn.Module,
    split_df: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    text_config = config["text_model"]
    dataset = TextNewsDataset(split_df, tokenizer=tokenizer, max_length=int(text_config["max_length"]))
    loader = DataLoader(
        dataset,
        batch_size=int(text_config["batch_size"]),
        shuffle=False,
        num_workers=int(text_config.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="text inference", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
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
                        "text_probability": float(probability),
                        "text_prediction": int(prediction),
                        "text_confidence": float(confidence),
                    }
                )
    return pd.DataFrame(rows)
