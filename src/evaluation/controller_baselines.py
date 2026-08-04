from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.evaluation.metrics import binary_classification_metrics
from src.fusion.state_builder import build_states
from src.utils.file_io import write_json
from src.utils.seed import set_seed


def run_controller_baselines(config: dict[str, Any]) -> dict[str, Any]:
    """Train simple learned controllers on the same state features as RL fusion.

    The output makes the RL-vs-simple-controller comparison explicit. Each model
    receives the identical reliability state used by the RL fusion controller.
    """
    seed = int(config.get("seed", 42))
    set_seed(seed)
    baseline_config = config["controller_baselines"]
    paths = config["paths"]
    feature_columns = baseline_config.get("features")

    train_df = pd.read_csv(paths["train_outputs"])
    validation_df = pd.read_csv(paths["validation_outputs"])
    test_df = pd.read_csv(paths["test_outputs"])
    train_x = build_states(train_df, feature_columns)
    train_y = train_df["label"].to_numpy(dtype=int)

    models = _build_models(baseline_config, seed)
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {"models": {}}
    predictions_dir = Path(paths.get("predictions_dir", "outputs/metrics/controller_baselines"))
    predictions_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        model.fit(train_x, train_y)
        validation_summary = _evaluate_model(
            model,
            validation_df,
            feature_columns,
            predictions_dir / f"{name}_validation_predictions.csv",
        )
        test_summary = _evaluate_model(
            model,
            test_df,
            feature_columns,
            predictions_dir / f"{name}_test_predictions.csv",
        )
        row = {
            "method": name,
            "state_dim": int(train_x.shape[1]),
            "validation_macro_f1": validation_summary["metrics"]["macro_f1"],
            "validation_accuracy": validation_summary["metrics"]["accuracy"],
            "validation_roc_auc": validation_summary["metrics"]["roc_auc"],
            "test_macro_f1": test_summary["metrics"]["macro_f1"],
            "test_accuracy": test_summary["metrics"]["accuracy"],
            "test_roc_auc": test_summary["metrics"]["roc_auc"],
            "test_predictions_path": test_summary["predictions_path"],
        }
        rows.append(row)
        details["models"][name] = {"validation": validation_summary, "test": test_summary}

    summary_path = Path(paths["summary_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("test_macro_f1", ascending=False).to_csv(summary_path, index=False)
    result = {
        "summary_path": str(summary_path),
        "details_path": paths["details_path"],
        "rows": rows,
        "comparison_note": "All controller baselines use the same reliability state features as the RL fusion controller.",
    }
    details["rows"] = rows
    write_json(details, paths["details_path"])
    return result


def _build_models(config: dict[str, Any], seed: int) -> dict[str, Any]:
    max_iter = int(config.get("logistic_max_iter", 1000))
    depths = [int(depth) for depth in config.get("tree_depths", [3, 5, None]) if depth is not None]
    models: dict[str, Any] = {
        "state_logistic_regression": LogisticRegression(max_iter=max_iter, random_state=seed),
    }
    for depth in depths:
        models[f"state_decision_tree_depth_{depth}"] = DecisionTreeClassifier(max_depth=depth, random_state=seed)
    if config.get("include_unlimited_tree", True):
        models["state_decision_tree_unlimited"] = DecisionTreeClassifier(random_state=seed)
    return models


def _evaluate_model(
    model: Any,
    df: pd.DataFrame,
    feature_columns: list[str] | None,
    output_path: Path,
) -> dict[str, Any]:
    x = build_states(df, feature_columns)
    labels = df["label"].to_numpy(dtype=int)
    probabilities = _positive_probabilities(model, x)
    predictions = (probabilities >= 0.5).astype(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "sample_id": df["sample_id"],
            "label": labels,
            "final_probability": probabilities,
            "final_prediction": predictions,
        }
    ).to_csv(output_path, index=False)
    return {
        "rows": int(len(df)),
        "metrics": binary_classification_metrics(labels, probabilities),
        "predictions_path": str(output_path),
    }


def _positive_probabilities(model: Any, x: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(x)
    class_index = list(model.classes_).index(1)
    return probabilities[:, class_index]
