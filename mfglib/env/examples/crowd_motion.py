from __future__ import annotations

import torch

from mfglib.env import Environment


class TransitionFn:
    def __init__(self, torus_l: int, torus_w: int, p_still: float) -> None:
        x = (1 - p_still) / 4
        eps_prob = [x, x, x, x, p_still]

        acts = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
        delta = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]

        self.p = torch.zeros(torus_l, torus_w, torus_l, torus_w, 5)
        for s_0 in range(torus_l):
            for s_1 in range(torus_w):
                for i, (a_0, a_1) in enumerate(acts):
                    for j, (dx, dy) in enumerate(delta):
                        s_next_0 = min(max(s_0 + a_0 + dx, 0), torus_l - 1)
                        s_next_1 = min(max(s_1 + a_1 + dy, 0), torus_w - 1)

                        self.p[s_next_0, s_next_1, s_0, s_1, i] += eps_prob[j]

    def __call__(self, env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor:
        return self.p


class RewardFn:
    def __init__(
        self,
        T: int,
        torus_l: int,
        torus_w: int,
        loc_change_freq: int,
        c: float,
        log_eps: float,
    ) -> None:
        self.c = c
        self.log_eps = log_eps

        bar_loc = (torus_l // 2, torus_w // 2)
        bar_locs = [bar_loc]

        for i in range(T + 1):
            if (i + 1) % loc_change_freq == 0:
                choices = [-1, 1]
                rand_rl = choices[torch.randint(0, 2, size=(1,))]
                rand_ud = choices[torch.randint(0, 2, size=(1,))]
                new_bar_loc = (bar_loc[0] + rand_rl, bar_loc[1] + rand_ud)
                if 0 <= new_bar_loc[0] < torus_l and 0 <= new_bar_loc[1] < torus_w:
                    bar_loc = new_bar_loc
            bar_locs.append(bar_loc)

        self.c1s = []
        for t in range(T + 1):
            bar_loc = bar_locs[t]
            dist = torch.zeros(torus_l, torus_w)
            for s_0 in range(torus_l):
                for s_1 in range(torus_w):
                    dist[s_0, s_1] = torch.abs(
                        torch.tensor(s_0 - bar_loc[0])
                    ) + torch.abs(torch.tensor(s_1 - bar_loc[1]))
            dist /= torus_l + torus_w
            c1 = 1.0 - dist.repeat(5, 1, 1).permute(1, 2, 0)
            self.c1s.append(c1)

    def __call__(self, env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor:
        mu_t = L_t.flatten(start_dim=2).sum(dim=-1)

        return self.c * self.c1s[t] - torch.log(
            mu_t.repeat(5, 1, 1).permute(1, 2, 0) + self.log_eps
        )
