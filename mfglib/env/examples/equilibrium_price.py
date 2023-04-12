from __future__ import annotations

from typing import cast

import torch

from mfglib.env import Environment


class TransitionFn:
    def __init__(self, s_inv: int, Q: int, H: int) -> None:
        self.p = torch.zeros(s_inv + 1, s_inv + 1, Q + 1, H + 1)
        for s in range(s_inv + 1):
            for q in range(Q + 1):
                for h in range(H + 1):
                    self.p[min(s - min(q, s) + h, s_inv), s, q, h] = 1.0

    def __call__(self, env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor:
        return self.p


class RewardFn:
    def __init__(
        self,
        s_inv: int,
        Q: int,
        H: int,
        d: float,
        e0: float,
        sigma: float,
        c: tuple[float, float, float, float, float],
    ) -> None:
        self.d = d
        self.e0 = e0
        self.sigma = sigma
        self.c = c

        self.s_tensor = torch.arange(s_inv + 1).repeat(Q + 1, H + 1, 1).permute(2, 0, 1)
        self.q_tensor = torch.arange(Q + 1).repeat(s_inv + 1, H + 1, 1).permute(0, 2, 1)
        self.h_tensor = torch.arange(H + 1).repeat(s_inv + 1, Q + 1, 1)

    def __call__(self, env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor:
        c0, c1, c2, c3, c4 = self.c

        p_t = (self.d / (L_t.mul(self.q_tensor).sum() + self.e0)) ** (1 / self.sigma)
        r_1 = (p_t - c0) * self.q_tensor
        r_2 = -c1 * self.q_tensor.pow(2)
        r_3 = -c2 * self.h_tensor
        r_4 = -(c2 + c3) * torch.maximum(self.q_tensor - self.s_tensor, torch.tensor(0))
        r_5 = -c4 * self.s_tensor
        r_t = r_1 + r_2 + r_3 + r_4 + r_5

        return cast(torch.Tensor, r_t)
