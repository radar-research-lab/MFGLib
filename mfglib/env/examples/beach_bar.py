from __future__ import annotations

import torch

from mfglib.env import Environment


class TransitionFn:
    def __init__(self, n: int, p_still: float) -> None:
        prs = [(1 - p_still) / 2, p_still, (1 - p_still) / 2]
        self.p = torch.zeros(n, n, 3)
        for s in range(n):
            for a in range(3):
                for epsilon, pr in zip([-1, 0, 1], prs):
                    s_next = min(max(s + a - 1 + epsilon, 0), n - 1)
                    self.p[s_next, s, a] += pr

    def __call__(self, env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor:
        return self.p


class RewardFn:
    def __init__(self, n: int, bar_loc: int, log_eps: float) -> None:
        self.c1 = torch.abs(
            torch.arange(0, n).repeat(3, 1).T - bar_loc * torch.ones(n, 3)
        )
        self.c2 = -torch.tensor([1, 0, 1]).repeat(n, 1) / n
        self.log_eps = log_eps

    def __call__(self, env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor:
        l_s = len(env.S)
        mu_t = L_t.flatten(start_dim=l_s).sum(-1)

        c3 = -torch.log(mu_t.repeat(L_t.shape[1], 1).T + self.log_eps) #- 1000 * L_t

        return self.c1 + self.c2 + c3
        # return -0.01*L_t
