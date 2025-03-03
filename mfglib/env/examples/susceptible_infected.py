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
        # NOTE:
        #         a = U, a = D
        #  s = S  [-1.0, -1.5]
        #  s = I  [ 0.0, -0.5]
        self.r = torch.tensor([[-1.0, -1.5], [0.0, -0.5]])

    def __call__(self, env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor:
        return self.r
