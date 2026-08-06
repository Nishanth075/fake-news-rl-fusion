from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import binary_classification_metrics
from src.fusion.q_network import FusionQNetwork
from src.fusion.state_builder import STATE_COLUMNS, build_states
from src.utils.seed import set_seed

REWARD_VARIANTS = ["binary", "negative_bce", "equal_relative_bce"]
TRIAL_ACTIONS = [(index / 10.0, 1.0 - index / 10.0) for index in range(11)]
SEEDS = [42, 7]
THRESHOLDS = np.arange(0.05, 0.951, 0.005)
EXPECTED_EQUAL_F1 = 0.8673332160630083


@dataclass
class PredictionInputs:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


class ContinuousFusionController(nn.Module):
    def __init__(self, state_dim: int = 9) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state).squeeze(1)


def run_rl_quick_trial(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = _default_config()
    if config:
        cfg = _deep_update(cfg, config)

    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = _load_and_verify_inputs(cfg)
    equal_summary = _evaluate_fixed_fusion(inputs.test, image_weight=0.5, threshold=0.5)
    if abs(equal_summary["macro_f1"] - EXPECTED_EQUAL_F1) > float(cfg["checks"]["equal_f1_tolerance"]):
        raise RuntimeError(
            "Equal fusion could not be reproduced from saved per-sample predictions. "
            f"Expected approximately {EXPECTED_EQUAL_F1}, got {equal_summary['macro_f1']}."
        )

    current_rl = _load_current_rl_summary(cfg)
    discrete_rows = _run_discrete_trials(inputs, cfg, equal_summary["macro_f1"], current_rl["macro_f1"])
    discrete_df = pd.DataFrame(discrete_rows)
    discrete_path = output_dir / "discrete_rl_trial.csv"
    discrete_df.to_csv(discrete_path, index=False)

    selected_reward = (
        discrete_df.groupby("reward_variant")["validation_macro_f1"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )
    best_discrete = discrete_df[discrete_df["reward_variant"] == selected_reward].copy()
    best_discrete["selected_by_validation_mean"] = True
    discrete_df["selected_by_validation_mean"] = discrete_df["reward_variant"] == selected_reward
    discrete_df.to_csv(discrete_path, index=False)

    continuous_rows = _run_continuous_trials(inputs, cfg, equal_summary["macro_f1"], current_rl["macro_f1"])
    continuous_df = pd.DataFrame(continuous_rows)
    continuous_path = output_dir / "continuous_controller_trial.csv"
    continuous_df.to_csv(continuous_path, index=False)

    comparison_rows = _comparison_rows(equal_summary, current_rl, best_discrete, continuous_df)
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = output_dir / "quick_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    conclusion_path = output_dir / "quick_conclusion.md"
    conclusion_path.write_text(
        _quick_conclusion(equal_summary, current_rl, best_discrete, continuous_df, selected_reward),
        encoding="utf-8",
    )
    summary = {
        "verification": {
            "train_rows": int(len(inputs.train)),
            "validation_rows": int(len(inputs.validation)),
            "test_rows": int(len(inputs.test)),
            "equal_fusion_test_macro_f1": equal_summary["macro_f1"],
        },
        "selected_discrete_reward": selected_reward,
        "created": {
            "discrete": str(discrete_path),
            "continuous": str(continuous_path),
            "comparison": str(comparison_path),
            "conclusion": str(conclusion_path),
        },
    }
    (output_dir / "quick_trial_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _run_discrete_trials(
    inputs: PredictionInputs,
    cfg: dict[str, Any],
    equal_macro_f1: float,
    current_rl_macro_f1: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    device = _device()
    fusion_cfg = cfg["discrete"]
    for reward_variant in REWARD_VARIANTS:
        for seed in SEEDS:
            set_seed(seed)
            model, validation_threshold, validation_metrics = _train_discrete_model(
                inputs.train, inputs.validation, fusion_cfg, reward_variant, seed, device
            )
            predictions = _predict_discrete(model, inputs.test, device)
            metrics = binary_classification_metrics(inputs.test["label"], predictions["final_probability"], validation_threshold)
            weights = predictions["image_weight"].to_numpy()
            rows.append(
                {
                    "experiment": "improved_discrete_rl",
                    "reward_variant": reward_variant,
                    "seed": seed,
                    "validation_threshold": validation_threshold,
                    "validation_macro_f1": validation_metrics["macro_f1"],
                    "test_macro_f1": metrics["macro_f1"],
                    "test_accuracy": metrics["accuracy"],
                    "test_roc_auc": metrics["roc_auc"],
                    "average_image_weight": float(weights.mean()),
                    "average_text_weight": float(1.0 - weights.mean()),
                    "action_distribution": json.dumps(_action_distribution(predictions["selected_action"], len(TRIAL_ACTIONS))),
                    "improvement_over_equal_fusion": float(metrics["macro_f1"] - equal_macro_f1),
                    "improvement_over_current_rl": float(metrics["macro_f1"] - current_rl_macro_f1),
                }
            )
    return rows


def _run_continuous_trials(
    inputs: PredictionInputs,
    cfg: dict[str, Any],
    equal_macro_f1: float,
    current_rl_macro_f1: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    device = _device()
    for seed in SEEDS:
        set_seed(seed)
        model, validation_threshold, validation_metrics = _train_continuous_model(
            inputs.train, inputs.validation, cfg["continuous"], seed, device
        )
        predictions = _predict_continuous(model, inputs.test, device)
        metrics = binary_classification_metrics(inputs.test["label"], predictions["final_probability"], validation_threshold)
        weights = predictions["image_weight"].to_numpy()
        rows.append(
            {
                "experiment": "continuous_controller",
                "seed": seed,
                "validation_threshold": validation_threshold,
                "validation_macro_f1": validation_metrics["macro_f1"],
                "test_macro_f1": metrics["macro_f1"],
                "test_accuracy": metrics["accuracy"],
                "test_roc_auc": metrics["roc_auc"],
                "average_image_weight": float(weights.mean()),
                "image_weight_std": float(weights.std(ddof=0)),
                "pct_weights_below_0_1": float((weights < 0.1).mean()),
                "pct_weights_above_0_9": float((weights > 0.9).mean()),
                "collapse_check": _collapse_check(weights),
                "improvement_over_equal_fusion": float(metrics["macro_f1"] - equal_macro_f1),
                "improvement_over_current_rl": float(metrics["macro_f1"] - current_rl_macro_f1),
            }
        )
    return rows


def _train_discrete_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    cfg: dict[str, Any],
    reward_variant: str,
    seed: int,
    device: torch.device,
) -> tuple[FusionQNetwork, float, dict[str, Any]]:
    states = torch.tensor(build_states(train_df, STATE_COLUMNS), dtype=torch.float32)
    dataset = TensorDataset(states, torch.arange(len(train_df), dtype=torch.long))
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=True, generator=generator)
    model = FusionQNetwork(
        state_dim=len(STATE_COLUMNS),
        action_dim=len(TRIAL_ACTIONS),
        dropout=float(cfg["dropout"]),
        hidden_dims=list(cfg["hidden_dims"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    criterion = nn.MSELoss()
    rng = np.random.default_rng(seed)
    best_state = copy.deepcopy(model.state_dict())
    best_threshold = 0.5
    best_metrics: dict[str, Any] = {"macro_f1": -1.0}
    epochs = int(cfg["epochs"])
    for epoch in range(1, epochs + 1):
        epsilon = _linear_epsilon(epoch, epochs, float(cfg["epsilon_start"]), float(cfg["epsilon_end"]))
        _train_discrete_epoch(model, loader, train_df, optimizer, criterion, device, epsilon, rng, reward_variant)
        val_predictions = _predict_discrete(model, validation_df, device)
        threshold, metrics = _select_threshold(validation_df["label"], val_predictions["final_probability"])
        if metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_metrics = metrics
            best_threshold = threshold
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_threshold, best_metrics


def _train_discrete_epoch(
    model: FusionQNetwork,
    loader: DataLoader,
    train_df: pd.DataFrame,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epsilon: float,
    rng: np.random.Generator,
    reward_variant: str,
) -> None:
    model.train()
    for states, indices in loader:
        states = states.to(device)
        indices_np = indices.numpy()
        with torch.no_grad():
            greedy_actions = torch.argmax(model(states), dim=1).detach().cpu().numpy()
        random_actions = rng.integers(0, len(TRIAL_ACTIONS), size=len(indices_np))
        actions_np = np.where(rng.random(len(indices_np)) < epsilon, random_actions, greedy_actions)
        rewards_np = _rewards(train_df.iloc[indices_np], actions_np, reward_variant)
        actions = torch.tensor(actions_np, dtype=torch.long, device=device).unsqueeze(1)
        rewards = torch.tensor(rewards_np, dtype=torch.float32, device=device)
        loss = criterion(model(states).gather(1, actions).squeeze(1), rewards)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def _train_continuous_model(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    cfg: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[ContinuousFusionController, float, dict[str, Any]]:
    train_states = torch.tensor(build_states(train_df, STATE_COLUMNS), dtype=torch.float32)
    train_labels = torch.tensor(train_df["label"].to_numpy(dtype=np.float32), dtype=torch.float32)
    dataset = TensorDataset(train_states, train_labels, torch.arange(len(train_df), dtype=torch.long))
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=True, generator=generator)
    model = ContinuousFusionController(state_dim=len(STATE_COLUMNS)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    best_state = copy.deepcopy(model.state_dict())
    best_threshold = 0.5
    best_metrics: dict[str, Any] = {"macro_f1": -1.0}
    epochs_without_improvement = 0
    for _epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        for states, labels, indices in loader:
            states = states.to(device)
            labels = labels.to(device)
            df_batch = train_df.iloc[indices.numpy()]
            image_probability = torch.tensor(df_batch["image_probability"].to_numpy(np.float32), device=device)
            text_probability = torch.tensor(df_batch["text_probability"].to_numpy(np.float32), device=device)
            image_weight = model(states)
            final_probability = image_weight * image_probability + (1.0 - image_weight) * text_probability
            loss = nn.functional.binary_cross_entropy(final_probability.clamp(1e-6, 1 - 1e-6), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        val_predictions = _predict_continuous(model, validation_df, device)
        threshold, metrics = _select_threshold(validation_df["label"], val_predictions["final_probability"])
        if metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_metrics = metrics
            best_threshold = threshold
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(cfg["early_stopping_patience"]):
            break
    model.load_state_dict(best_state)
    return model, best_threshold, best_metrics


def _predict_discrete(model: FusionQNetwork, df: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    model.eval()
    states = torch.tensor(build_states(df, STATE_COLUMNS), dtype=torch.float32, device=device)
    with torch.no_grad():
        actions = torch.argmax(model(states), dim=1).detach().cpu().numpy().astype(int)
    weights = np.asarray([TRIAL_ACTIONS[action][0] for action in actions], dtype=np.float32)
    return _prediction_frame(df, weights, actions)


def _predict_continuous(model: ContinuousFusionController, df: pd.DataFrame, device: torch.device) -> pd.DataFrame:
    model.eval()
    states = torch.tensor(build_states(df, STATE_COLUMNS), dtype=torch.float32, device=device)
    with torch.no_grad():
        weights = model(states).detach().cpu().numpy().astype(np.float32)
    return _prediction_frame(df, weights, None)


def _prediction_frame(df: pd.DataFrame, image_weights: np.ndarray, actions: np.ndarray | None) -> pd.DataFrame:
    final_probability = (
        image_weights * df["image_probability"].to_numpy(dtype=np.float32)
        + (1.0 - image_weights) * df["text_probability"].to_numpy(dtype=np.float32)
    )
    frame = pd.DataFrame(
        {
            "sample_id": df["sample_id"],
            "label": df["label"].astype(int),
            "image_weight": image_weights,
            "text_weight": 1.0 - image_weights,
            "final_probability": final_probability,
        }
    )
    if actions is not None:
        frame["selected_action"] = actions
    return frame


def _rewards(df: pd.DataFrame, actions: np.ndarray, reward_variant: str) -> np.ndarray:
    image_weights = np.asarray([TRIAL_ACTIONS[action][0] for action in actions], dtype=np.float32)
    y_true = df["label"].to_numpy(dtype=np.float32)
    fused = _fused_probability(df, image_weights)
    if reward_variant == "binary":
        return np.where((fused >= 0.5).astype(int) == y_true.astype(int), 1.0, -1.0).astype(np.float32)
    fused_bce = _bce(y_true, fused)
    if reward_variant == "negative_bce":
        return (-fused_bce).astype(np.float32)
    if reward_variant == "equal_relative_bce":
        equal_bce = _bce(y_true, _fused_probability(df, np.full(len(df), 0.5, dtype=np.float32)))
        return (equal_bce - fused_bce).astype(np.float32)
    raise ValueError(f"Unknown reward variant: {reward_variant}")


def _bce(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(probabilities.astype(np.float64), 1e-6, 1 - 1e-6)
    return -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))


def _fused_probability(df: pd.DataFrame, image_weights: np.ndarray) -> np.ndarray:
    return (
        image_weights * df["image_probability"].to_numpy(dtype=np.float32)
        + (1.0 - image_weights) * df["text_probability"].to_numpy(dtype=np.float32)
    )


def _select_threshold(labels: pd.Series | np.ndarray, probabilities: pd.Series | np.ndarray) -> tuple[float, dict[str, Any]]:
    best_threshold = 0.5
    best_metrics: dict[str, Any] = {"macro_f1": -1.0}
    for threshold in THRESHOLDS:
        metrics = binary_classification_metrics(labels, probabilities, float(threshold))
        if metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_metrics = metrics
            best_threshold = float(threshold)
    return best_threshold, best_metrics


def _evaluate_fixed_fusion(df: pd.DataFrame, image_weight: float, threshold: float) -> dict[str, Any]:
    probability = _fused_probability(df, np.full(len(df), image_weight, dtype=np.float32))
    return binary_classification_metrics(df["label"], probability, threshold)


def _load_and_verify_inputs(cfg: dict[str, Any]) -> PredictionInputs:
    paths = cfg["paths"]
    frames = {
        "train": _read_required_csv(paths["train_outputs"]),
        "validation": _read_required_csv(paths["validation_outputs"]),
        "test": _read_required_csv(paths["test_outputs"]),
    }
    required_columns = {
        "sample_id",
        "label",
        "image_probability",
        "image_confidence",
        "image_quality",
        "text_probability",
        "text_confidence",
        "text_quality",
    }
    for split, frame in frames.items():
        missing = sorted(required_columns - set(frame.columns))
        if missing:
            raise ValueError(f"{split} output file is missing required columns: {missing}")
        if frame["sample_id"].isna().any() or frame["label"].isna().any():
            raise ValueError(f"{split} has missing sample_id or label values")
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"{split} has duplicate sample_id values")
        bad_labels = set(frame["label"].astype(int).unique()) - {0, 1}
        if bad_labels:
            raise ValueError(f"{split} has labels outside {{0, 1}}: {sorted(bad_labels)}")
    overlaps = {
        "train_validation": set(frames["train"]["sample_id"]) & set(frames["validation"]["sample_id"]),
        "train_test": set(frames["train"]["sample_id"]) & set(frames["test"]["sample_id"]),
        "validation_test": set(frames["validation"]["sample_id"]) & set(frames["test"]["sample_id"]),
    }
    non_empty = {name: len(values) for name, values in overlaps.items() if values}
    if non_empty:
        raise ValueError(f"Cross-split sample_id overlap detected: {non_empty}")
    return PredictionInputs(frames["train"], frames["validation"], frames["test"])


def _read_required_csv(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required saved prediction file not found: {file_path}. "
            "Run this in the Colab/Drive copy that contains data/final_modality_outputs/*.csv."
        )
    return pd.read_csv(file_path)


def _load_current_rl_summary(cfg: dict[str, Any]) -> dict[str, float]:
    path = Path(cfg["paths"]["current_rl_metrics"])
    if not path.exists():
        raise FileNotFoundError(f"Current RL metrics file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data["test"]["metrics"]
    return {
        "macro_f1": float(metrics["macro_f1"]),
        "accuracy": float(metrics["accuracy"]),
        "roc_auc": float(metrics["roc_auc"]),
    }


def _comparison_rows(
    equal_summary: dict[str, Any],
    current_rl: dict[str, float],
    best_discrete: pd.DataFrame,
    continuous_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    return [
        {
            "method": "equal_fusion",
            "validation_macro_f1": None,
            "test_macro_f1_mean": equal_summary["macro_f1"],
            "test_accuracy_mean": equal_summary["accuracy"],
            "test_roc_auc_mean": equal_summary["roc_auc"],
            "improvement_over_equal_fusion_mean": 0.0,
            "improvement_over_current_rl_mean": equal_summary["macro_f1"] - current_rl["macro_f1"],
        },
        {
            "method": "current_discrete_rl",
            "validation_macro_f1": None,
            "test_macro_f1_mean": current_rl["macro_f1"],
            "test_accuracy_mean": current_rl["accuracy"],
            "test_roc_auc_mean": current_rl["roc_auc"],
            "improvement_over_equal_fusion_mean": current_rl["macro_f1"] - equal_summary["macro_f1"],
            "improvement_over_current_rl_mean": 0.0,
        },
        _aggregate_row("best_improved_discrete_rl", best_discrete),
        _aggregate_row("continuous_controller", continuous_df),
    ]


def _aggregate_row(method: str, df: pd.DataFrame) -> dict[str, Any]:
    return {
        "method": method,
        "validation_macro_f1": float(df["validation_macro_f1"].mean()),
        "test_macro_f1_mean": float(df["test_macro_f1"].mean()),
        "test_accuracy_mean": float(df["test_accuracy"].mean()),
        "test_roc_auc_mean": float(df["test_roc_auc"].mean()),
        "test_macro_f1_std": float(df["test_macro_f1"].std(ddof=0)),
        "improvement_over_equal_fusion_mean": float(df["improvement_over_equal_fusion"].mean()),
        "improvement_over_current_rl_mean": float(df["improvement_over_current_rl"].mean()),
    }


def _quick_conclusion(
    equal_summary: dict[str, Any],
    current_rl: dict[str, float],
    best_discrete: pd.DataFrame,
    continuous_df: pd.DataFrame,
    selected_reward: str,
) -> str:
    equal_f1 = float(equal_summary["macro_f1"])
    current_f1 = float(current_rl["macro_f1"])
    discrete_deltas = best_discrete["test_macro_f1"] - equal_f1
    continuous_deltas = continuous_df["test_macro_f1"] - equal_f1
    discrete_mean = float(best_discrete["test_macro_f1"].mean())
    continuous_mean = float(continuous_df["test_macro_f1"].mean())
    discrete_same_direction = bool((discrete_deltas > 0).all() or (discrete_deltas < 0).all())
    continuous_same_direction = bool((continuous_deltas > 0).all() or (continuous_deltas < 0).all())
    continuous_collapse = ", ".join(sorted(set(continuous_df["collapse_check"].astype(str))))
    continue_full = (
        (discrete_mean >= 0.871 and (discrete_deltas > 0).all())
        or (continuous_mean >= 0.871 and (continuous_deltas > 0).all())
    )
    return "\n".join(
        [
            "# Quick RL-Only Feasibility Conclusion",
            "",
            f"- Equal fusion test macro F1: {equal_f1:.12f}",
            f"- Current RL test macro F1: {current_f1:.12f}",
            f"- Selected improved discrete reward: `{selected_reward}`",
            f"- Best improved discrete RL mean test macro F1 across seeds: {discrete_mean:.12f}",
            f"- Continuous controller mean test macro F1 across seeds: {continuous_mean:.12f}",
            "",
            "## Decision Questions",
            "",
            f"1. Did improved discrete RL beat equal fusion? {'Yes' if (discrete_deltas > 0).all() else 'No'}",
            f"2. Did the continuous controller beat equal fusion? {'Yes' if (continuous_deltas > 0).all() else 'No'}",
            f"3. Did both seeds show the same direction? Discrete: {'Yes' if discrete_same_direction else 'No'}; Continuous: {'Yes' if continuous_same_direction else 'No'}",
            f"4. Was the gain larger than the current 0.0003 macro-F1 improvement? Discrete: {'Yes' if discrete_mean - equal_f1 > 0.0003 else 'No'}; Continuous: {'Yes' if continuous_mean - equal_f1 > 0.0003 else 'No'}",
            f"5. Did either controller collapse? Continuous collapse check: {continuous_collapse}. Discrete action distributions are reported in `discrete_rl_trial.csv`.",
            f"6. Is the result promising enough to continue with full experiments? {'Yes' if continue_full else 'No'}",
            "",
            "## Rule Applied",
            "",
            "Continue full experiments only if at least one controller reaches around 0.871 or above and both seeds improve in the same direction.",
        ]
    )


def _action_distribution(actions: pd.Series | np.ndarray, action_count: int) -> dict[str, int]:
    action_array = np.asarray(actions).astype(int)
    return {str(index): int((action_array == index).sum()) for index in range(action_count)}


def _collapse_check(weights: np.ndarray) -> str:
    mean = float(weights.mean())
    std = float(weights.std(ddof=0))
    if mean < 0.1:
        return "text_only_collapse"
    if mean > 0.9:
        return "image_only_collapse"
    if abs(mean - 0.5) < 0.03 and std < 0.03:
        return "equal_fusion_collapse"
    return "not_collapsed"


def _linear_epsilon(epoch: int, epochs: int, start: float, end: float) -> float:
    if epochs <= 1:
        return end
    return float(start + ((epoch - 1) / (epochs - 1)) * (end - start))


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(updated.get(key), dict):
            updated[key] = _deep_update(updated[key], value)
        else:
            updated[key] = value
    return updated


def _default_config() -> dict[str, Any]:
    return {
        "paths": {
            "train_outputs": "data/final_modality_outputs/train_outputs.csv",
            "validation_outputs": "data/final_modality_outputs/validation_outputs.csv",
            "test_outputs": "data/final_modality_outputs/test_outputs.csv",
            "current_rl_metrics": "outputs/metrics/final_rl_fusion_metrics.json",
            "output_dir": "outputs/rl_quick_trial",
        },
        "checks": {
            "equal_f1_tolerance": 0.001,
        },
        "discrete": {
            "hidden_dims": [64, 32],
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 64,
            "epochs": 50,
            "epsilon_start": 1.0,
            "epsilon_end": 0.05,
        },
        "continuous": {
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 64,
            "epochs": 100,
            "early_stopping_patience": 10,
        },
    }
