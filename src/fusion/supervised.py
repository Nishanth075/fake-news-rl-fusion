from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import binary_classification_metrics
from src.fusion.state_builder import build_states
from src.utils.device import get_device
from src.utils.file_io import write_json
from src.utils.seed import set_seed


class SupervisedFusionMLP(nn.Module):
    """Small supervised classifier over image-text reliability state features."""

    def __init__(self, state_dim: int = 9, hidden_dims: list[int] | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]
        layers: list[nn.Module] = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(1)


def train_supervised_fusion(config: dict[str, Any]) -> dict[str, Any]:
    """Train a supervised MLP fusion baseline on the same reliability state as RL."""
    seed = int(config.get("seed", 42))
    set_seed(seed)
    model_config = config["supervised_fusion"]
    paths = config["paths"]
    device = get_device()

    train_df = pd.read_csv(paths["train_outputs"])
    validation_df = pd.read_csv(paths["validation_outputs"])
    feature_columns = model_config.get("features")

    train_states = torch.tensor(build_states(train_df, feature_columns), dtype=torch.float32)
    train_labels = torch.tensor(train_df["label"].to_numpy(dtype="float32"), dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(train_states, train_labels),
        batch_size=int(model_config["batch_size"]),
        shuffle=True,
    )

    model = SupervisedFusionMLP(
        state_dim=int(model_config.get("state_dim", train_states.shape[1])),
        hidden_dims=list(model_config.get("hidden_dims", [64, 32])),
        dropout=float(model_config.get("dropout", 0.1)),
    ).to(device)
    model.feature_columns = feature_columns
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config.get("weight_decay", 0.0)),
    )
    criterion = nn.BCEWithLogitsLoss()

    checkpoint_dir = Path(paths["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_supervised_fusion_mlp.pt"
    best_f1 = -1.0
    history = []

    for epoch in range(1, int(model_config["epochs"]) + 1):
        train_loss = _train_one_epoch(model, loader, optimizer, criterion, device)
        validation_predictions = predict_supervised_fusion(model, validation_df, device)
        validation_metrics = binary_classification_metrics(
            validation_df["label"].to_numpy(), validation_predictions["final_probability"].to_numpy()
        )
        record = {"epoch": epoch, "train_loss": train_loss, "validation_metrics": validation_metrics}
        history.append(record)
        macro_f1 = float(validation_metrics["macro_f1"])
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "validation_macro_f1": best_f1,
                },
                checkpoint_path,
            )

    validation_summary = evaluate_supervised_fusion(config, split="validation")
    test_summary = evaluate_supervised_fusion(config, split="test")
    result = {
        "device": str(device),
        "best_validation_macro_f1": best_f1,
        "best_checkpoint_path": str(checkpoint_path),
        "history": history,
        "validation": validation_summary,
        "test": test_summary,
    }
    write_json(result, paths["metrics_path"])
    return result


def evaluate_supervised_fusion(config: dict[str, Any], split: str = "test") -> dict[str, Any]:
    paths = config["paths"]
    model_config = config["supervised_fusion"]
    device = get_device()
    df = pd.read_csv(paths[f"{split}_outputs"])
    checkpoint_path = Path(paths["checkpoint_dir"]) / "best_supervised_fusion_mlp.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Supervised fusion checkpoint not found: {checkpoint_path}")

    feature_columns = model_config.get("features")
    model = SupervisedFusionMLP(
        state_dim=int(model_config.get("state_dim", len(feature_columns or []))),
        hidden_dims=list(model_config.get("hidden_dims", [64, 32])),
        dropout=float(model_config.get("dropout", 0.1)),
    ).to(device)
    model.feature_columns = feature_columns
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions = predict_supervised_fusion(model, df, device)
    metrics = binary_classification_metrics(df["label"].to_numpy(), predictions["final_probability"].to_numpy())

    output_path = Path(paths.get(f"{split}_predictions_path", paths["test_predictions_path"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    return {
        "split": split,
        "rows": int(len(predictions)),
        "metrics": metrics,
        "predictions_path": str(output_path),
    }


def predict_supervised_fusion(model: nn.Module, df: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    model.eval()
    feature_columns = getattr(model, "feature_columns", None)
    states = torch.tensor(build_states(df, feature_columns), dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(states)
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    return pd.DataFrame(
        {
            "sample_id": df["sample_id"],
            "label": df["label"].astype(int),
            "final_probability": probabilities,
            "final_prediction": (probabilities >= 0.5).astype(int),
        }
    )


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for states, labels in loader:
        states = states.to(device)
        labels = labels.to(device)
        logits = model(states)
        loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch_size = len(labels)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size
    return total_loss / max(total_examples, 1)

