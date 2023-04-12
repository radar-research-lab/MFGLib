from __future__ import annotations

import torch

from mfglib.env import Environment


class TransitionFn:
    def __init__(self) -> None:
        self.p = torch.zeros(4, 4, 3)
        for s in range(4):
            for a in range(3):
                self.p[1 + a, s, a] = 1.0

    def __call__(self, env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor:
        return self.p


class RewardFn:
    def __call__(self, env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor:
        mu_t = L_t.sum(dim=1)

        r_t = torch.zeros(4, 3)
        r_t[1, :] = torch.ones(3).mul(2 * mu_t[3] - mu_t[2])
        r_t[2, :] = torch.ones(3).mul(4 * mu_t[1] - 2 * mu_t[3])
        r_t[3, :] = torch.ones(3).mul(6 * mu_t[2] - 3 * mu_t[1])

        return r_t
