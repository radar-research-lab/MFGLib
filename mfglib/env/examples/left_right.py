from __future__ import annotations

import torch

from mfglib.env import Environment


class TransitionFn:
    def __init__(self) -> None:
        self.p = torch.zeros(3, 3, 2)
        for s in range(3):
            for a in range(2):
                self.p[1 + a, s, a] = 1.0

    def __call__(self, env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor:
        return self.p


class RewardFn:
    def __call__(self, env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor:
        mu_t = L_t.sum(dim=1)

        r_t = torch.zeros(env.S + env.A)
        r_t[1, :] = -torch.ones(env.A).mul(mu_t[1])
        r_t[2, :] = -torch.ones(env.A).mul(2 * mu_t[2])

        return r_t
