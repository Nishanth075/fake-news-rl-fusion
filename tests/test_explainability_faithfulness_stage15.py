from __future__ import annotations

from src.explainability.faithfulness import _aggregate, _delete_token_indices


def test_delete_token_indices_removes_salient_tokens() -> None:
    text = _delete_token_indices(["fake", "news", "image", "claim"], {1, 3})

    assert text == "fake image"


def test_faithfulness_aggregate_reports_salient_and_random_means() -> None:
    aggregate = _aggregate(
        [
            {
                "image_salient_comprehensiveness": 0.2,
                "image_random_comprehensiveness": 0.05,
                "text_salient_comprehensiveness": 0.3,
                "text_random_comprehensiveness": 0.1,
            },
            {
                "image_salient_comprehensiveness": 0.4,
                "image_random_comprehensiveness": 0.15,
                "text_salient_comprehensiveness": 0.5,
                "text_random_comprehensiveness": 0.2,
            },
        ]
    )

    assert aggregate["image_salient_comprehensiveness_mean"] == 0.30000000000000004
    assert aggregate["text_random_comprehensiveness_mean"] == 0.15000000000000002
