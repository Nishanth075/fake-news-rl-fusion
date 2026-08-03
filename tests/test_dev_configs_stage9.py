from __future__ import annotations

from pathlib import Path

from src.utils.config import load_yaml


DEV_CONFIGS = [
    "configs/dev_subset.yaml",
    "configs/dev_image_download.yaml",
    "configs/dev_image_model.yaml",
    "configs/dev_text_model.yaml",
    "configs/dev_reliability.yaml",
    "configs/dev_baselines.yaml",
    "configs/dev_fusion.yaml",
    "configs/dev_ablation.yaml",
    "configs/dev_rl_analysis.yaml",
    "configs/dev_robustness.yaml",
    "configs/dev_results_export.yaml",
]


def test_dev_configs_load() -> None:
    for config_path in DEV_CONFIGS:
        config = load_yaml(config_path)
        assert config, config_path


def test_dev_configs_do_not_reuse_debug_output_dirs() -> None:
    image_config = load_yaml("configs/dev_image_model.yaml")
    text_config = load_yaml("configs/dev_text_model.yaml")
    reliability_config = load_yaml("configs/dev_reliability.yaml")

    assert Path(image_config["paths"]["train_csv"]).parts[1] == "dev_splits_available"
    assert Path(text_config["paths"]["train_csv"]).parts[1] == "dev_splits_available"
    assert Path(reliability_config["reliability"]["output_dir"]).parts[1] == "dev_modality_outputs"



