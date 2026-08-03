from __future__ import annotations

from typing import Any

FUSION_ACTIONS: list[tuple[float, float]] = [
    (0.90, 0.10),
    (0.75, 0.25),
    (0.60, 0.40),
    (0.50, 0.50),
    (0.40, 0.60),
    (0.25, 0.75),
    (0.10, 0.90),
]


def resolve_fusion_actions(config: dict[str, Any] | None = None) -> list[tuple[float, float]]:
    if not config:
        return FUSION_ACTIONS
    configured_actions = config.get("fusion", {}).get("action_weights")
    if not configured_actions:
        return FUSION_ACTIONS
    actions = [(float(image_weight), float(text_weight)) for image_weight, text_weight in configured_actions]
    for image_weight, text_weight in actions:
        if abs((image_weight + text_weight) - 1.0) > 1e-6:
            raise ValueError(f"Fusion action weights must sum to 1.0: {(image_weight, text_weight)}")
    return actions


def get_action_weights(action_index: int, actions: list[tuple[float, float]] | None = None) -> tuple[float, float]:
    action_space = actions or FUSION_ACTIONS
    if action_index < 0 or action_index >= len(action_space):
        raise ValueError(f"Invalid fusion action index: {action_index}")
    return action_space[action_index]
