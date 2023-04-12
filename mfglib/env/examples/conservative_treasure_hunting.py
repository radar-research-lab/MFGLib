from __future__ import annotations

import torch

from mfglib.env import Environment


class TransitionFn:
    def __init__(self, n: int, c: tuple[float, ...]) -> None:
        self.n = n
        self.c = c

    def __call__(self, env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor:
        p_t = torch.zeros(self.n, self.n, self.n)

        if t == 0:
            for s in range(self.n):
                for a in range(self.n):
                    p_t[a, s, a] = 1.0
            return p_t

        for s in range(self.n):
            eye_s = torch.zeros(self.n, self.n)
            eye_s[s, s] = 1.0
            diff_s = (L_t - eye_s).pow(2).sum()
            p_t[:, s, :] = (
                (self.c[t - 1] * diff_s)
                / (1.0 + self.n * self.c[t - 1] * diff_s)
                * torch.ones(self.n, self.n)
            )
            for a in range(self.n):
                p_t[a, s, a] += 1.0 / (1.0 + self.n * self.c[t - 1] * diff_s)

        return p_t


class RewardFn:
    def __init__(self, n: int, r: tuple[float, ...]) -> None:
        self.n = n
        self.r = r

    def __call__(self, env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor:
        r_t = torch.zeros(self.n, self.n)

        if t == 0:
            return r_t

        for s in range(self.n):
            eye_s = torch.zeros(self.n, self.n)
            eye_s[s, s] = 1.0
            diff_s = (L_t - eye_s).pow(2).sum() / 2.0
            r_t[s, s] = self.r[s] * (1 - diff_s)

        return r_t
