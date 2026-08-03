from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import binary_classification_metrics
from src.fusion.actions import FUSION_ACTIONS
from src.fusion.q_network import FusionQNetwork
from src.fusion.reward import reward_from_predictions
from src.fusion.state_builder import build_states
from src.utils.device import get_device
from src.utils.file_io import write_json
from src.utils.seed import set_seed


def train_rl_fusion(config: dict[str, Any]) -> dict[str, Any]:
    """Train an offline contextual-bandit fusion controller."""
    seed = int(config.get("seed", 42))
    set_seed(seed)
    fusion_config = config["fusion"]
    paths = config["paths"]
    device = get_device()

    train_df = pd.read_csv(paths["train_outputs"])
    validation_df = pd.read_csv(paths["validation_outputs"])
    feature_columns = fusion_config.get("features")
    train_states = torch.tensor(build_states(train_df, feature_columns), dtype=torch.float32)
    train_dataset = TensorDataset(train_states, torch.arange(len(train_df), dtype=torch.long))
    loader = DataLoader(
        train_dataset,
        batch_size=int(fusion_config["batch_size"]),
        shuffle=True,
    )

    model = FusionQNetwork(
        state_dim=int(fusion_config.get("state_dim", len(feature_columns or []))),
        action_dim=int(fusion_config["action_dim"]),
        dropout=float(fusion_config.get("dropout", 0.1)),
    ).to(device)
    model.feature_columns = feature_columns
    optimizer = torch.optim.Adam(model.parameters(), lr=float(fusion_config["learning_rate"]))
    criterion = nn.MSELoss()

    checkpoint_dir = Path(paths["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_fusion_q_network.pt"

    epochs = int(fusion_config["epochs"])
    epsilon_start = float(fusion_config["epsilon_start"])
    epsilon_end = float(fusion_config["epsilon_end"])
    best_f1 = -1.0
    history = []

    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        epsilon = _linear_epsilon(epoch, epochs, epsilon_start, epsilon_end)
        train_loss, train_reward = _train_one_epoch(
            model, loader, train_df, optimizer, criterion, device, epsilon, rng
        )
        validation_predictions = predict_with_model(model, validation_df, device)
        validation_metrics = binary_classification_metrics(
            validation_df["label"].to_numpy(), validation_predictions["final_probability"].to_numpy()
        )
        record = {
            "epoch": epoch,
            "epsilon": epsilon,
            "train_loss": train_loss,
            "train_mean_reward": train_reward,
            "validation_metrics": validation_metrics,
            "action_distribution": _action_distribution(validation_predictions["selected_action"].to_numpy()),
            "average_image_weight": float(validation_predictions["image_weight"].mean()),
            "average_text_weight": float(validation_predictions["text_weight"].mean()),
        }
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

    validation_summary = evaluate_rl_fusion(config, split="validation")
    test_summary = evaluate_rl_fusion(config, split="test")
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


def evaluate_rl_fusion(config: dict[str, Any], split: str = "test") -> dict[str, Any]:
    paths = config["paths"]
    device = get_device()
    split_path = paths[f"{split}_outputs"]
    df = pd.read_csv(split_path)
    checkpoint_path = Path(paths["checkpoint_dir"]) / "best_fusion_q_network.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Fusion checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    feature_columns = config["fusion"].get("features")
    model = FusionQNetwork(
        state_dim=int(config["fusion"].get("state_dim", len(feature_columns or []))),
        action_dim=int(config["fusion"]["action_dim"]),
        dropout=float(config["fusion"].get("dropout", 0.1)),
    ).to(device)
    model.feature_columns = feature_columns
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions = predict_with_model(model, df, device)
    metrics = binary_classification_metrics(df["label"].to_numpy(), predictions["final_probability"].to_numpy())

    output_path = Path(paths.get(f"{split}_predictions_path", paths.get("test_predictions_path", f"outputs/metrics/rl_fusion_{split}_predictions.csv")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    return {
        "split": split,
        "rows": int(len(predictions)),
        "metrics": metrics,
        "action_distribution": _action_distribution(predictions["selected_action"].to_numpy()),
        "average_image_weight": float(predictions["image_weight"].mean()),
        "average_text_weight": float(predictions["text_weight"].mean()),
        "predictions_path": str(output_path),
    }


def predict_with_model(model: nn.Module, df: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    model.eval()
    feature_columns = getattr(model, "feature_columns", None)
    states = torch.tensor(build_states(df, feature_columns), dtype=torch.float32, device=device)
    with torch.no_grad():
        actions = torch.argmax(model(states), dim=1).detach().cpu().numpy().astype(int)
    weights = np.asarray([FUSION_ACTIONS[action] for action in actions], dtype=np.float32)
    final_probability = (
        weights[:, 0] * df["image_probability"].to_numpy(dtype=np.float32)
        + weights[:, 1] * df["text_probability"].to_numpy(dtype=np.float32)
    )
    return pd.DataFrame(
        {
            "sample_id": df["sample_id"],
            "label": df["label"].astype(int),
            "selected_action": actions,
            "image_weight": weights[:, 0],
            "text_weight": weights[:, 1],
            "final_probability": final_probability,
            "final_prediction": (final_probability >= 0.5).astype(int),
        }
    )


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    train_df: pd.DataFrame,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epsilon: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_reward = 0.0
    total_examples = 0
    action_count = len(FUSION_ACTIONS)
    for states, indices in loader:
        states = states.to(device)
        indices_np = indices.numpy()
        with torch.no_grad():
            greedy_actions = torch.argmax(model(states), dim=1).detach().cpu().numpy()
        random_actions = rng.integers(0, action_count, size=len(indices_np))
        explore = rng.random(len(indices_np)) < epsilon
        actions_np = np.where(explore, random_actions, greedy_actions).astype(int)
        rewards_np = _calculate_rewards(train_df.iloc[indices_np], actions_np)

        actions = torch.tensor(actions_np, dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards_np, dtype=torch.float32, device=device)
        q_values = model(states).gather(1, actions).squeeze(1)
        loss = criterion(q_values, rewards)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = len(indices_np)
        total_loss += float(loss.item()) * batch_size
        total_reward += float(rewards_np.sum())
        total_examples += batch_size
    return total_loss / max(total_examples, 1), total_reward / max(total_examples, 1)


def _calculate_rewards(df: pd.DataFrame, actions: np.ndarray) -> np.ndarray:
    weights = np.asarray([FUSION_ACTIONS[action] for action in actions], dtype=np.float32)
    final_probability = (
        weights[:, 0] * df["image_probability"].to_numpy(dtype=np.float32)
        + weights[:, 1] * df["text_probability"].to_numpy(dtype=np.float32)
    )
    predictions = (final_probability >= 0.5).astype(int)
    return reward_from_predictions(predictions, df["label"].to_numpy(dtype=int))


def _linear_epsilon(epoch: int, epochs: int, start: float, end: float) -> float:
    if epochs <= 1:
        return end
    progress = (epoch - 1) / (epochs - 1)
    return float(start + progress * (end - start))


def _action_distribution(actions: np.ndarray) -> dict[str, int]:
    return {str(index): int((actions == index).sum()) for index in range(len(FUSION_ACTIONS))}


