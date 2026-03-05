from __future__ import annotations

import time
from typing import Any, Callable, TypedDict

import torch.nn

from mfglib.env import Environment
from mfglib.scoring import exploitability_score as expl_score
from mfglib.utils import mean_field_from_policy, policy_from_mean_field


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
            d_shaped = d.reshape(self.env.T + 1, *self.env.S, *self.env.A)
            return (
                -self.λ[0] * self.social_reward(d_shaped)
                + self.λ[1] * torch.einsum("tsa,tsa->", z, d)
                + self.ρ[0] * self.g(d)
                + self.ρ[1] * self.h(y, z, d)
            )

        def g(self, d: torch.Tensor) -> torch.Tensor:
            T = self.env.T
            n_states = self.env.n_states
            n_actions = self.env.n_actions

            err = torch.empty(T + 1, n_states)
            for t in range(T):
                d_t = d[t].reshape(*self.env.S, *self.env.A)
                P_t = self.env.prob(t, d_t).reshape(n_states, n_states, n_actions)
                term_1 = torch.einsum("jsa,sa->j", P_t, d[t])
                term_2 = torch.einsum("ja->j", d[t + 1])
                err[t] = term_1 - term_2
            err[T] = torch.einsum("sa->s", d[0]) - self.env.mu0.reshape(n_states)
            return err.square().sum()

        def h(self, y: torch.Tensor, z: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
            T = self.env.T
            n_states = self.env.n_states
            n_actions = self.env.n_actions

            err = torch.empty(T + 1, n_states, n_actions, dtype=torch.double)
            Z = (
                torch.eye(n_states, dtype=torch.double)
                .unsqueeze(dim=1)
                .repeat(1, n_actions, 1)
            )
            d_0 = d[0].reshape(*self.env.S, *self.env.A)
            c_0 = -self.env.reward(0, d_0).reshape(n_states, n_actions)
            P_0 = self.env.prob(0, d_0).reshape(n_states, n_states, n_actions)
            term_1 = torch.einsum("jsa,j->sa", P_0, y[0])
            term_2 = torch.einsum("saj,j->sa", Z, y[T])
            err[0] = term_1 + term_2 + z[0] - c_0
            for t in range(1, T):
                d_t = d[t].reshape(*self.env.S, *self.env.A)
                c_t = -self.env.reward(t, d_t).reshape(n_states, n_actions)
                P_t = self.env.prob(t, d_t).reshape(n_states, n_states, n_actions)
                term_1 = torch.einsum("saj,j->sa", -Z, y[t - 1])
                term_2 = torch.einsum("jsa,j->sa", P_t, y[t])
                err[t] = term_1 + term_2 + z[t] - c_t
            d_T = d[T].reshape(*self.env.S, *self.env.A)
            c_T = -self.env.reward(T, d_T).reshape(n_states, n_actions)
            term_1 = torch.einsum("saj,j->sa", -Z, y[T - 1])
            err[T] = term_1 + z[T] - c_T
            return err.square().sum()

    def __init__(
        self, λ: tuple[float, float] = (0, 1), ρ: tuple[float, float] = (1, 1)
    ) -> None:
        self.λ = λ
        self.ρ = ρ

    def solve(
        self,
        env: Environment,
        social_reward: Callable[[torch.Tensor], float] | None = None,
        max_iter: int = 100,
        kwargs: MESOB.Kwargs | None = None,
        atol: float | None = None,
    ) -> dict[str, torch.Tensor]:
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

        d = kwargs.pop("d0", default_d0).to(dtype=torch.double)
        y = kwargs.pop("y0", default_y0).to(dtype=torch.double)
        z = kwargs.pop("z0", default_z0).to(dtype=torch.double)

        d.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)

        obj = self.Objective(env, social_reward, self.λ, self.ρ)

        opt_cls = kwargs.pop("opt_cls", torch.optim.Adam)
        opt_kwargs = kwargs.pop("opt_kwargs", {})
        opt = opt_cls([d, y, z], **opt_kwargs)

        y_radius = n_states * (T + 1) * (T + 2) * r_max / 2
        z_radius = n_states * n_actions * (T**2 + T + 2) * r_max

        expl = torch.empty(max_iter, dtype=torch.double)
        welf = torch.empty(max_iter, dtype=torch.double)
        grad_norm = torch.empty(max_iter, dtype=torch.double)
        solve_time = torch.empty(max_iter, dtype=torch.double)

        min_grad_norm = float("inf")
        d_best = d.detach().clone()
        y_best = y.detach().clone()
        z_best = z.detach().clone()

        t0 = time.perf_counter()
        for i in range(max_iter):
            with torch.no_grad():
                d_prev = d.clone()
                z_prev = z.clone()
                y_prev = y.clone()

            opt.zero_grad()
            obj_val = obj(d, y, z)
            obj_val.backward()
            opt.step()

            with torch.no_grad():
                d_next = d.flatten(start_dim=1)
                d_next = project_simplex(d_next)
                d_next = d_next.reshape(d.shape)

                y_next = project_l2_ball(y, radius=y_radius)
                z_next = project_Z_set(z, radius=z_radius)

                # _assert_dyz_is_feasible(
                #     d=d_next, y=y_next, z=z_next, y_radius=y_radius, z_radius=z_radius
                # )

                Δd_norm = torch.linalg.vector_norm(d_next - d_prev)
                Δy_norm = torch.linalg.vector_norm(y_next - y_prev)
                Δz_norm = torch.linalg.vector_norm(z_next - z_prev)

                d_shaped = d_next.reshape(T + 1, *env.S, *env.A)
                pi = policy_from_mean_field(d_shaped, env=env)
                d_pi = mean_field_from_policy(pi, env=env)

                grad_norm[i] = (
                    torch.sqrt(Δd_norm**2 + Δy_norm**2 + Δz_norm**2) / opt_kwargs["lr"]
                )
                expl[i] = expl_score(env, pi)
                welf[i] = social_reward(d_pi)
                solve_time[i] = time.perf_counter() - t0

                d.copy_(d_next)
                y.copy_(y_next)
                z.copy_(z_next)

                if grad_norm[i] < min_grad_norm:
                    min_grad_norm = grad_norm[i]
                    d_best = d_next.detach().clone()
                    y_best = y_next.detach().clone()
                    z_best = z_next.detach().clone()

                if atol is not None and grad_norm[i] < atol:
                    break

        return dict(
            d=d_best,
            y=y_best,
            z=z_best,
            grad_norm=grad_norm[: i + 1],
            expl=expl[: i + 1],
            welf=welf[: i + 1],
            solve_time=solve_time[: i + 1],
        )


def project_simplex(tensor, dim=-1, radius=1.0):
    sorted_tensor, _ = torch.sort(tensor, descending=True, dim=dim)
    cumsum_tensor = torch.cumsum(sorted_tensor, dim=dim)

    shape = list(tensor.shape)
    n_features = shape[dim]

    indices = torch.arange(1, n_features + 1, dtype=tensor.dtype)
    view_shape = [1] * len(shape)
    view_shape[dim] = n_features
    indices = indices.view(*view_shape)

    threshold_candidates = (cumsum_tensor - radius) / indices

    mask = sorted_tensor > threshold_candidates
    rho = torch.sum(mask, dim=dim, keepdim=True)

    rho_indices = (rho - 1).long()
    theta = torch.gather(threshold_candidates, dim, rho_indices)

    projected_tensor = torch.relu(tensor - theta)

    return projected_tensor


def project_l2_ball(v: torch.Tensor, *, radius: float = 1.0) -> torch.Tensor:
    v_norm = v.norm()
    if v_norm <= radius:
        return v
    else:
        return v / v_norm * radius  # type: ignore[no-any-return]


def project_Z_set(z: torch.Tensor, *, radius: float) -> torch.Tensor:
    z_clamp = z.clamp(min=0)
    if z_clamp.sum() > radius:
        z_clamp = z.flatten(start_dim=1)
        z_clamp = project_simplex(z_clamp, radius=radius)
    return z_clamp.reshape(z.shape)


def _assert_dyz_is_feasible(
    *,
    d: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    y_radius: float,
    z_radius: float,
    eps: float = 1e-10,
) -> None:
    assert torch.all(d >= -eps) and torch.all(d <= 1 + eps), f"{d.min()}, {d.max()}"
    dvec = d.flatten(start_dim=1).sum(dim=1)
    assert torch.allclose(dvec, torch.ones_like(dvec))

    yvec = y.flatten(start_dim=1).norm(p=2, dim=1)
    assert torch.all(yvec <= y_radius)

    assert torch.all(z >= 0)
    zvec = z.flatten(start_dim=1).sum(dim=1)
    assert torch.all(zvec <= z_radius + eps), (zvec - z_radius).max()
