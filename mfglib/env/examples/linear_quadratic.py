from __future__ import annotations

import math
from typing import cast

import torch

from mfglib.env import Environment


class TransitionFn:
    def __init__(self, el: int, m: int, sigma: float, delta: float, k: float) -> None:
        self.el = el
        self.m = m
        self.sigma = sigma
        self.delta = delta
        self.k = k

        self.epsilon = torch.arange(-3, 4) * sigma
        self.epsilon_pr = torch.exp(-0.5 * self.epsilon.pow(2)) / torch.sqrt(
            torch.tensor(2 * math.pi)
        )
        self.epsilon_pr /= self.epsilon_pr.sum()

    def __call__(self, env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor:
        mu_t = L_t.flatten(start_dim=1).sum(dim=-1)
        m_t = torch.arange(-self.el, self.el + 1).mul(mu_t).sum()

        n_s = 2 * self.el + 1
        n_a = 2 * self.m + 1

        p_t = torch.zeros(n_s, n_s, n_a)
        for s in range(n_s):
            for a in range(n_a):
                for epsilon, epsilon_pr in zip(self.epsilon, self.epsilon_pr):
                    s_next_tensor = (
                        s
                        - self.el
                        + (self.k * (m_t - s + self.el) + a - self.m) * self.delta
                        + self.sigma * epsilon * math.sqrt(self.delta)
                    )
                    s_next = cast(int, torch.round(s_next_tensor).int().item())
                    s_next = min(max(-self.el, s_next), self.el)
                    p_t[s_next + self.el, s, a] += epsilon_pr

        return p_t


class RewardFn:
    def __init__(
        self,
        T: int,
        el: int,
        m: int,
        delta: float,
        q: float,
        kappa: float,
        c_term: float,
    ) -> None:
        self.T = T
        self.el = el
        self.delta = delta
        self.kappa = kappa
        self.c_term = c_term

        self.r_c1 = -0.5 * torch.arange(-m, m + 1).pow(2).repeat(2 * el + 1, 1)
        self.r_c21 = q * torch.arange(-m, m + 1).repeat(2 * el + 1, 1)
        self.r_c22 = torch.arange(-el, el + 1).repeat(2 * m + 1, 1).T

    def __call__(self, env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor:
        mu_t = L_t.sum(dim=-1)
        m_t = torch.arange(-self.el, self.el + 1).mul(mu_t).sum()

        if t == self.T:
            return -0.5 * self.c_term * (m_t - self.r_c22).pow(2)
        else:
            r_t = (
                self.r_c1
                + self.r_c21.mul(m_t - self.r_c22)
                - 0.5 * self.kappa * (m_t - self.r_c22).pow(2)
            )
            return r_t * self.delta
