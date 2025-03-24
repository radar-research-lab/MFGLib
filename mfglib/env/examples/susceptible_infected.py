from __future__ import annotations

import torch

from mfglib.env import Environment


class TransitionFn:
    def __call__(self, env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor:
        mu_t = L_t.sum(dim=1)

        p_t = torch.zeros(2, 2, 2)
        for s in range(2):
            for a in range(2):
                if s == 1:
                    p_t[0, s, a] = 0.3
                    p_t[1, s, a] = 0.7
                elif a == 0:
                    p_t[1, s, a] = (0.9**2) * mu_t[1]
                    p_t[0, s, a] = 1.0 - (0.9**2) * mu_t[1]
                else:
                    p_t[0, s, a] = 1.0

        return p_t


class RewardFn:
    def __init__(self) -> None:
        self.r = torch.zeros(2, 2)
        self.r[0, 1] = -0.5
        self.r[1, 0] = -1.0
        self.r[1, 1] = -1.5

    def __call__(self, env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor:
        return self.r
