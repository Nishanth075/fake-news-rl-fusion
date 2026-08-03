from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import binary_classification_metrics
from src.fusion.actions import resolve_fusion_actions
from src.fusion.q_network import FusionQNetwork
from src.fusion.train import predict_with_model
from src.utils.device import get_device
from src.utils.file_io import write_json
from src.utils.seed import set_seed


def run_robustness(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate how the trained fusion policy reacts to controlled modality degradation."""
    seed = int(config.get("seed", 42))
    set_seed(seed)
    rng = np.random.default_rng(seed)

    paths = config["paths"]
    output_dir = Path(paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model = _load_model(config, device)
    original_df = pd.read_csv(paths["test_outputs"])
    original_predictions = predict_with_model(model, original_df, device)

    rows = []
    details: dict[str, Any] = {"original": _summarize_predictions(original_df, original_predictions)}
    for corruption_name, corruption_config in config["corruptions"].items():
        modality = str(corruption_config["modality"])
        for level in corruption_config["levels"]:
            level_float = float(level)
            corrupted_df = _apply_corruption(original_df, corruption_name, modality, level_float, rng)
            corrupted_predictions = predict_with_model(model, corrupted_df, device)
            summary = _summarize_delta(
                original_df,
                corrupted_df,
                original_predictions,
                corrupted_predictions,
                modality,
            )
            summary.update({"corruption": corruption_name, "modality": modality, "level": level_float})
            rows.append(summary)
            details[f"{corruption_name}_{level_float}"] = summary

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(paths["summary_path"], index=False)
    write_json(details, paths["details_path"])
    return {"summary_path": str(paths["summary_path"]), "rows": rows}


def _load_model(config: dict[str, Any], device: torch.device) -> FusionQNetwork:
    checkpoint_path = Path(config["paths"]["checkpoint_dir"]) / "best_fusion_q_network.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Fusion checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("config", config)
    fusion_config = model_config["fusion"]
    feature_columns = fusion_config.get("features")
    action_weights = resolve_fusion_actions(model_config)
    model = FusionQNetwork(
        state_dim=int(fusion_config.get("state_dim", len(feature_columns or []))),
        action_dim=len(action_weights),
        dropout=float(fusion_config.get("dropout", 0.1)),
        hidden_dims=fusion_config.get("hidden_dims"),
    ).to(device)
    model.feature_columns = feature_columns
    model.action_weights = action_weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _apply_corruption(
    df: pd.DataFrame,
    corruption_name: str,
    modality: str,
    level: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    corrupted = df.copy()
    if corruption_name.endswith("quality_drop"):
        quality_col = f"{modality}_quality"
        corrupted[quality_col] = (corrupted[quality_col] * (1.0 - level)).clip(0, 1)
    elif corruption_name.endswith("probability_noise"):
        prob_col = f"{modality}_probability"
        conf_col = f"{modality}_confidence"
        noise = rng.normal(loc=0.0, scale=level, size=len(corrupted))
        corrupted[prob_col] = (corrupted[prob_col] + noise).clip(0, 1)
        corrupted[f"{modality}_prediction"] = (corrupted[prob_col] >= 0.5).astype(int)
        corrupted[conf_col] = np.maximum(corrupted[prob_col], 1.0 - corrupted[prob_col])
    else:
        raise ValueError(f"Unknown corruption type: {corruption_name}")
    return corrupted


def _summarize_predictions(df: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, Any]:
    metrics = binary_classification_metrics(df["label"].to_numpy(), predictions["final_probability"].to_numpy())
    return {
        "metrics": metrics,
        "average_image_weight": float(predictions["image_weight"].mean()),
        "average_text_weight": float(predictions["text_weight"].mean()),
    }


def _summarize_delta(
    original_df: pd.DataFrame,
    corrupted_df: pd.DataFrame,
    original_predictions: pd.DataFrame,
    corrupted_predictions: pd.DataFrame,
    modality: str,
) -> dict[str, Any]:
    metrics = binary_classification_metrics(
        corrupted_df["label"].to_numpy(), corrupted_predictions["final_probability"].to_numpy()
    )
    quality_delta = float((corrupted_df[f"{modality}_quality"] - original_df[f"{modality}_quality"]).mean())
    image_weight_delta = float(
        (corrupted_predictions["image_weight"] - original_predictions["image_weight"]).mean()
    )
    text_weight_delta = float(
        (corrupted_predictions["text_weight"] - original_predictions["text_weight"]).mean()
    )
    target_weight_col = f"{modality}_weight"
    target_weight_delta = image_weight_delta if modality == "image" else text_weight_delta
    return {
        "rows": int(len(corrupted_df)),
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
        "roc_auc": metrics["roc_auc"],
        "average_image_weight": float(corrupted_predictions["image_weight"].mean()),
        "average_text_weight": float(corrupted_predictions["text_weight"].mean()),
        "delta_image_weight": image_weight_delta,
        "delta_text_weight": text_weight_delta,
        "delta_quality": quality_delta,
        "target_weight": target_weight_col,
        "delta_target_weight": target_weight_delta,
        "adapted_in_expected_direction": bool(target_weight_delta < 0),
    }
