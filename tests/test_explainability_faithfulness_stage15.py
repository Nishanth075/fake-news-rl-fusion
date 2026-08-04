from __future__ import annotations

import torch

from src.explainability.faithfulness import _aggregate, _delete_token_indices, _score_text_masked_probability


def test_delete_token_indices_removes_salient_tokens() -> None:
    text = _delete_token_indices(["fake", "news", "image", "claim"], {1, 3})

    assert text == "fake image"


def test_score_text_masked_probability_masks_original_token_ids() -> None:
    class DummyTokenizer:
        mask_token_id = 103
        unk_token_id = 100

    class DummyModel:
        def __call__(self, input_ids, attention_mask):
            assert input_ids.tolist()[0][1] == 103
            assert input_ids.tolist()[0][3] == 103
            return {"logits": torch.tensor([[0.0, 2.0]])}

    probability = _score_text_masked_probability(
        DummyModel(),
        torch.tensor([101, 10, 11, 12, 102]),
        torch.tensor([1, 1, 1, 1, 1]),
        DummyTokenizer(),
        {1, 3},
        torch.device("cpu"),
        1,
    )

    assert probability > 0.8


def test_faithfulness_aggregate_reports_salient_and_random_means() -> None:
    aggregate = _aggregate(
        [
            {
                "image_salient_comprehensiveness": 0.2,
                "image_random_comprehensiveness": 0.05,
                "text_salient_comprehensiveness": 0.3,
                "text_least_comprehensiveness": 0.02,
                "text_random_comprehensiveness": 0.1,
            },
            {
                "image_salient_comprehensiveness": 0.4,
                "image_random_comprehensiveness": 0.15,
                "text_salient_comprehensiveness": 0.5,
                "text_least_comprehensiveness": 0.04,
                "text_random_comprehensiveness": 0.2,
            },
        ]
    )

    assert aggregate["image_salient_comprehensiveness_mean"] == 0.30000000000000004
    assert aggregate["text_least_comprehensiveness_mean"] == 0.03
    assert aggregate["text_random_comprehensiveness_mean"] == 0.15000000000000002
