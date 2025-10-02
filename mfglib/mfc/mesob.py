from __future__ import annotations

from typing import Any, Callable, TypedDict

import torch.nn

from mfglib.env import Environment
from mfglib.scoring import exploitability_score as expl_score


class MESOB:

    class Kwargs(TypedDict, total=False):
        """MESOB-specific keyword arguments passed into solve()."""

        opt_cls: type[torch.optim.Optimizer]
        opt_kwargs: dict[str, Any]
        y0: torch.Tensor
        z0: torch.Tensor
        d0: torch.Tensor

    class Objective(torch.nn.Module):
        """MESOB objective function."""

        def __init__(
            self,
            env: Environment,
            social_reward: Callable[[torch.Tensor], float],
            λ: tuple[float, float],
            ρ: tuple[float, float],
        ) -> None:
            super().__init__()
            self.env = env
            self.social_reward = social_reward
            self.λ = λ
            self.ρ = ρ

        def forward(
            self, d: torch.Tensor, y: torch.Tensor, z: torch.Tensor
        ) -> torch.Tensor:
            return (
                -self.λ[0] * self.social_reward(d)
                + self.λ[1] * torch.einsum("tsa,tsa->", z, d)
                + self.ρ[0] * self.g(d)
                + self.ρ[1] * self.h(y, z, d)
            )

        def g(self, d: torch.Tensor) -> torch.Tensor:
            err = torch.empty(self.env.T + 1, self.env.n_states)
            for t in range(self.env.T):
                P_t = self.env.prob(t, d[t])
                term_1 = torch.einsum("jsa,sa->j", P_t, d[t])
                term_2 = torch.einsum("ja->j", d[t + 1])
                err[t] = term_1 - term_2
            err[self.env.T] = torch.einsum("sa->s", d[0]) - self.env.mu0
            return err.square().sum()

        def h(self, y: torch.Tensor, z: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
            T = self.env.T
            n_states = self.env.n_states
            n_actions = self.env.n_actions
            err = torch.empty(T + 1, n_states, n_actions)
            Z = torch.eye(n_states).unsqueeze(dim=1).repeat(1, n_actions, 1)
            c_0 = -self.env.reward(0, d[0])
            P_0 = self.env.prob(0, d[0])
            term_1 = torch.einsum("jsa,j->sa", P_0, y[0])
            term_2 = torch.einsum("saj,j->sa", Z, y[T])
            err[0] = term_1 + term_2 + z[0] - c_0
            for t in range(1, T):
                c_t = -self.env.reward(t, d[t])
                P_t = self.env.prob(t, d[t])
                term_1 = torch.einsum("saj,j->sa", -Z, y[t - 1])
                term_2 = torch.einsum("jsa,j->sa", P_t, y[t])
                err[t] = term_1 + term_2 + z[t] - c_t
            c_T = -self.env.reward(T, d[T])
            term_1 = torch.einsum("saj,j->sa", -Z, y[T - 1])
            err[T] = term_1 + z[T] - c_T
            return err.square().sum()

    def __init__(
        self, λ: tuple[float, float], ρ: tuple[float, float] = (1.0, 1.0)
    ) -> None:
        self.λ = λ
        self.ρ = ρ

    def solve(
        self,
        env: Environment,
        social_reward: Callable[[torch.Tensor], float] | None = None,
        max_iter: int = 100,
        kwargs: MESOB.Kwargs | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """TODO -- what is the best way to specify social metrics and weights?"""
        if social_reward is None:
            social_reward = lambda _: 0.0
        kwargs = kwargs or {}

        T = env.T
        n_states = env.n_states
        n_actions = env.n_actions
        r_max = env.r_max

        default_d0 = torch.ones(T + 1, n_states, n_actions) / n_states / n_actions
        default_y0 = torch.zeros(T + 1, n_states)
        default_z0 = torch.zeros(T + 1, n_states, n_actions)

        d = kwargs.pop("d0", default_d0)
        y = kwargs.pop("y0", default_y0)
        z = kwargs.pop("z0", default_z0)

        d.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)

        obj = self.Objective(env, social_reward, self.λ, self.ρ)

        opt_cls = kwargs.pop("opt_cls", torch.optim.Adam)
        opt_kwargs = kwargs.pop("opt_kwargs", {})
        opt = opt_cls([d, y, z], **opt_kwargs)

        y_radius = n_states * (T + 1) * (T + 2) * r_max / 2
        z_radius = n_states * n_actions * (T**2 + T + 2) * r_max

        pis = torch.empty(max_iter + 1, T + 1, n_states, n_actions)
        expls = torch.empty(max_iter + 1)
        obj_vals = torch.empty(max_iter + 1)

        for i in range(max_iter):
            obj_val = obj(d, y, z)

            pi = d / d.sum(dim=-1, keepdim=True)
            pi = pi.nan_to_num(nan=1 / n_actions)

            pis[i] = pi.data.clone()
            expls[i] = expl_score(env, pi)
            obj_vals[i] = obj_val.data.clone()

            opt.zero_grad()
            obj_val.backward()
            opt.step()

            d.data = project_simplex(d.data, axis=0)
            y.data = project_l2_ball(y.data, r=y_radius)
            z.data = project_Z_set(z.data, r=z_radius)

        obj_val = obj(d, y, z)

        pi = d / d.sum(dim=-1, keepdim=True)
        pi = pi.nan_to_num(nan=1 / n_actions)

        pis[max_iter] = pi.data.clone()
        expls[max_iter] = expl_score(env, pi)
        obj_vals[max_iter] = obj_val.data.clone()

        return pis, expls, obj_vals


def project_simplex(v: torch.Tensor, r: float = 1.0, axis: int = -1) -> torch.Tensor:
    """TODO.
    Args:
        v:
        r:
        axis:
    Returns:
    """
    if r == 0.0:
        return torch.zeros_like(v)

    def project_simplex_2d(v_2d: torch.Tensor) -> torch.Tensor:
        """Helper function.
        Args:
            v_2d: (M, N)-shaped collection of vectors. Each row is interpreted
                as an N-dimensional vector.
        Returns:
            (M, N)-shaped tensor with each row residing in the z-simplex.
        """
        M, N = v_2d.shape
        mu = torch.sort(v_2d, dim=1, descending=True)[0]
        cumsum = torch.cumsum(mu, dim=1)
        j = torch.arange(1, N + 1).repeat(M, 1)
        ρ = torch.sum(mu * j - cumsum + r > 0.0, dim=1, keepdim=True) - 1
        theta = (cumsum.gather(1, ρ) - r) / (ρ + 1)
        return torch.clamp(v_2d - theta, min=0.0)

    shape = v.shape

    if len(shape) == 1:
        v_2d = torch.unsqueeze(v, 0)
        return project_simplex_2d(v_2d)[0]
    else:
        axis = axis % len(shape)
        v_2d = v.movedim(axis, 0).flatten(start_dim=1)
        w_2d = project_simplex_2d(v_2d)
        w_2d = w_2d.reshape([shape[axis], *shape[1:axis], *shape[axis + 1 :]])
        return w_2d.movedim(axis, 0)


def project_l2_ball(v: torch.Tensor, *, r: float = 1.0) -> torch.Tensor:
    v_norm = v.norm()
    if v_norm <= r:
        return v
    else:
        return v / v_norm * r  # type: ignore[no-any-return]


def project_Z_set(z: torch.Tensor, *, r: float) -> torch.Tensor:
    z_clamp = z.ravel().clamp(min=0)
    if z_clamp.sum() > r:
        z_clamp = project_simplex(z_clamp, r=r)
    return z_clamp.reshape(z.shape)
