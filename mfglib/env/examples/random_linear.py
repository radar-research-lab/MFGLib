from __future__ import annotations

from typing import cast

import torch

from mfglib import __TORCH_FLOAT__

from mfglib.env import Environment


class TransitionFn:
    def __init__(self, n: int, m: float) -> None:
        self.p1 = 2 * m * torch.rand(n, n, n, dtype=torch.float64 if __TORCH_FLOAT__ == 64 else torch.float) - m
        self.p2 = 2 * m * torch.rand(n, n, n, dtype=torch.float64 if __TORCH_FLOAT__ == 64 else torch.float) - m

    def __call__(self, env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor:
        soft_max = torch.nn.Softmax(dim=0)
        return cast(torch.Tensor, soft_max(self.p1 @ (L_t.double() if __TORCH_FLOAT__ == 64 else L_t.float()) + self.p2))


class RewardFn:
    def __init__(self, n: int, m: float) -> None:
        self.r1 = 2 * m * torch.rand(n, n, dtype=torch.float64 if __TORCH_FLOAT__ == 64 else torch.float) - m
        self.r2 = 2 * m * torch.rand(n, n, dtype=torch.float64 if __TORCH_FLOAT__ == 64 else torch.float) - m

    def __call__(self, env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor:
        return self.r1 @ (L_t.double() if __TORCH_FLOAT__ == 64 else L_t.float()) + self.r2
