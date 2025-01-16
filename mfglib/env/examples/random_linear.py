from __future__ import annotations

from typing import cast

import torch

from mfglib.env import Environment


class TransitionFn:
    def __init__(self, n: int, m: float) -> None:
        self.p1 = 2 * m * torch.rand(n, n, n) - m
        self.p2 = 2 * m * torch.rand(n, n, n) - m

    def __call__(self, env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor:
        soft_max = torch.nn.Softmax(dim=0)
        return cast(
            torch.Tensor,
            soft_max(self.p1 @ L_t + self.p2),
        )


class RewardFn:
    def __init__(self, n: int, m: float) -> None:
        self.r1 = 2 * m * torch.rand(n, n) - m
        self.r2 = 2 * m * torch.rand(n, n) - m

    def __call__(self, env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor:
        return self.r1 @ L_t + self.r2
