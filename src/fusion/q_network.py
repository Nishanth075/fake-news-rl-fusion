from __future__ import annotations

import torch
from torch import nn


class FusionQNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int = 9,
        action_dim: int = 7,
        dropout: float = 0.1,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        dims = [state_dim, *(hidden_dims or [64, 32]), action_dim]
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend([nn.Linear(input_dim, output_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
