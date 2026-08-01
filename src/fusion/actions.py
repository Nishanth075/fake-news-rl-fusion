from __future__ import annotations

FUSION_ACTIONS: list[tuple[float, float]] = [
    (0.90, 0.10),
    (0.75, 0.25),
    (0.60, 0.40),
    (0.50, 0.50),
    (0.40, 0.60),
    (0.25, 0.75),
    (0.10, 0.90),
]


def get_action_weights(action_index: int) -> tuple[float, float]:
    if action_index < 0 or action_index >= len(FUSION_ACTIONS):
        raise ValueError(f"Invalid fusion action index: {action_index}")
    return FUSION_ACTIONS[action_index]
