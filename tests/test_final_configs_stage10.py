from __future__ import annotations

from pathlib import Path

from src.utils.config import load_yaml


FINAL_CONFIGS = [
    "configs/final_subset.yaml",
    "configs/final_image_download.yaml",
    "configs/final_image_model.yaml",
    "configs/final_text_model.yaml",
    "configs/final_reliability.yaml",
    "configs/final_baselines.yaml",
    "configs/final_fusion.yaml",
    "configs/final_ablation.yaml",
    "configs/final_rl_analysis.yaml",
    "configs/final_robustness.yaml",
    "configs/final_results_export.yaml",
]


def test_final_configs_load() -> None:
    for config_path in FINAL_CONFIGS:
        config = load_yaml(config_path)
        assert config, config_path


def test_final_configs_use_isolated_outputs() -> None:
    subset_config = load_yaml("configs/final_subset.yaml")
    image_config = load_yaml("configs/final_image_model.yaml")
    text_config = load_yaml("configs/final_text_model.yaml")
    reliability_config = load_yaml("configs/final_reliability.yaml")
    fusion_config = load_yaml("configs/final_fusion.yaml")

    assert Path(subset_config["subset"]["output_dir"]).parts[1] == "final_splits"
    assert Path(image_config["paths"]["train_csv"]).parts[1] == "final_splits_available"
    assert Path(text_config["paths"]["train_csv"]).parts[1] == "final_splits_available"
    assert Path(reliability_config["reliability"]["output_dir"]).parts[1] == "final_modality_outputs"
    assert "final_fusion" in fusion_config["paths"]["checkpoint_dir"]


def test_final_subset_size_is_larger_than_dev_but_bounded() -> None:
    final_config = load_yaml("configs/final_subset.yaml")
    subset = final_config["subset"]

    assert subset["train_size"] == 20000
    assert subset["validation_size"] == 4000
    assert subset["test_size"] == 4000


