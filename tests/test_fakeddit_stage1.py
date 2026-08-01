from __future__ import annotations

import pandas as pd

from src.data.fakeddit import prepare_fakeddit


def test_prepare_fakeddit_converts_official_splits(tmp_path) -> None:
    raw_dir = tmp_path / "raw" / "fakeddit_v2" / "multimodal_only_samples"
    raw_dir.mkdir(parents=True)

    sample = pd.DataFrame(
        {
            "id": ["real_1", "fake_1"],
            "clean_title": ["real post", "fake post"],
            "image_url": ["https://example.com/real.jpg", "https://example.com/fake.jpg"],
            "2_way_label": [1, 0],
            "hasImage": [True, True],
        }
    )
    sample.to_csv(raw_dir / "multimodal_train.tsv", sep="\t", index=False)
    sample.to_csv(raw_dir / "multimodal_validate.tsv", sep="\t", index=False)
    sample.to_csv(raw_dir / "multimodal_test_public.tsv", sep="\t", index=False)

    config = {
        "seed": 42,
        "fakeddit": {
            "raw_dir": str(raw_dir),
            "train_tsv": "multimodal_train.tsv",
            "validation_tsv": "multimodal_validate.tsv",
            "test_tsv": "multimodal_test_public.tsv",
            "output_csv": str(tmp_path / "processed" / "fakeddit_dataset.csv"),
            "splits_dir": str(tmp_path / "splits"),
            "stats_path": str(tmp_path / "metrics" / "fakeddit_stats.json"),
            "image_output_dir": "data/images/fakeddit",
            "image_extension": ".jpg",
            "invert_2_way_label": True,
            "validation": {
                "min_text_chars": 1,
                "remove_duplicate_text": False,
                "remove_duplicate_image_paths": False,
            },
        },
    }

    stats = prepare_fakeddit(config)
    train = pd.read_csv(tmp_path / "splits" / "train.csv")

    assert stats["total_rows"] == 6
    assert train.loc[train["sample_id"] == "real_1", "label"].item() == 0
    assert train.loc[train["sample_id"] == "fake_1", "label"].item() == 1
    assert (tmp_path / "processed" / "fakeddit_dataset.csv").exists()
    assert (tmp_path / "metrics" / "fakeddit_stats.json").exists()
