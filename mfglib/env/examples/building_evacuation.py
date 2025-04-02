from __future__ import annotations

import torch

from mfglib.env import Environment


class TransitionFn:
    def __init__(self, n_floor: int, floor_l: int, floor_w: int) -> None:
        S = (n_floor, floor_l, floor_w)

        self.p = torch.zeros(S + S + (6,))
        for f in range(n_floor):
            for l_tile in range(floor_l):
                for w_tile in range(floor_w):
                    # a = 0: going right
                    l_tile_new = min(l_tile + 1, floor_l - 1)
                    self.p[f, l_tile_new, w_tile, f, l_tile, w_tile, 0] = 1.0

                    # a = 1: going left
                    l_tile_new = max(0, l_tile - 1)
                    self.p[f, l_tile_new, w_tile, f, l_tile, w_tile, 1] = 1.0

                    # a = 2: going up
                    w_tile_new = max(0, w_tile - 1)
                    self.p[f, l_tile, w_tile_new, f, l_tile, w_tile, 2] = 1.0

                    # a = 3: going down
                    w_tile_new = min(w_tile + 1, floor_w - 1)
                    self.p[f, l_tile, w_tile_new, f, l_tile, w_tile, 3] = 1.0

                    # a = 4: going downstairs
                    f_new = max(0, f - 1) if l_tile == 0 and w_tile == 0 else f
                    self.p[f_new, l_tile, w_tile, f, l_tile, w_tile, 4] = 1.0

                    # a = 5: going upstairs
                    f_new = (
                        min(f + 1, n_floor - 1)
                        if l_tile == floor_l - 1 and w_tile == floor_w - 1
                        else f
                    )
                    self.p[f_new, l_tile, w_tile, f, l_tile, w_tile, 5] = 1.0

    def __call__(self, env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor:
        return self.p


class RewardFn:
    def __init__(
        self, S: tuple[int, int, int], eta: float, log_eps: float, evac_r: float
    ) -> None:
        self.eta = eta
        self.log_eps = log_eps

        self.r = torch.zeros(S)
        self.r[0] = evac_r

    def __call__(self, env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor:
        mu_t = L_t.sum(dim=-1).clamp(min=self.log_eps)  # prevent ZeroDivisionError

        reward = -self.eta * torch.log(mu_t) + self.r

        # Passing -1 as the size for a dimension means not changing the size of that dimension.
        return reward.unsqueeze(dim=-1).expand(-1, -1, -1, 6)
