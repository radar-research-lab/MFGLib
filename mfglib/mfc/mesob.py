from __future__ import annotations

from typing import Callable, TypedDict

import torch.nn


class Environment:
    T: int
    S: int
    A: int
    r_max: int
    mu0: torch.Tensor

    def r(self, t: int, L_t: torch.Tensor) -> torch.Tensor:
        """Returned reward vector is (S, A)-shaped."""
        raise NotImplementedError

    def P(self, t: int, L_t: torch.Tensor) -> torch.Tensor:
        """Returned transition kernel is (S, A, S)-shaped."""
        raise NotImplementedError


class MESOB:

    class Kwargs(TypedDict, total=False):
        """MESOB-specific keyword arguments passed into solve()."""

        opt_cls: type[torch.optim.Optimizer]
        opt_kwargs: dict[str, float]
        y0: torch.Tensor
        z0: torch.Tensor
        d0: torch.Tensor

    class Objective(torch.nn.Module):
        """MESOB objective function."""

        def __init__(
            self,
            env: Environment,
            social_weights: list[float],
            social_metrics: list[Callable[[torch.Tensor], float]],
            w: float,
            rho: tuple[float, float],
        ) -> None:
            super().__init__()
            self.env = env
            self.social_weights = torch.tensor(social_weights)
            self.social_metrics = social_metrics
            self.w = w
            self.rho = rho

        def forward(
            self, d: torch.Tensor, y: torch.Tensor, z: torch.Tensor
        ) -> torch.Tensor:
            return (
                -sum(
                    wgt * V(d)
                    for wgt, V in zip(self.social_weights, self.social_metrics)
                )
                + self.w * torch.einsum("tsa,tsa->", z, d)
                + self.rho[0] * self.g(d)
                + self.rho[1] * self.h(y, z, d)
            )

        def g(self, d: torch.Tensor) -> torch.Tensor:
            err = torch.empty([self.env.T + 1, self.env.S])
            for t in range(self.env.T):
                P_t = self.env.P(t, d[t])
                for s in range(self.env.S):
                    LHS = torch.einsum("sa,sa->", P_t[:, :, s], d[t])
                    RHS = d[t + 1, s].sum()
                    err[t, s] = LHS - RHS
            err[self.env.T] = d[0].sum(dim=-1) - self.env.mu0
            return err.square().sum()

        def h(self, y: torch.Tensor, z: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
            err = torch.empty([self.env.T + 1, self.env.S, self.env.A])
            Z = torch.eye(self.env.S).tile(self.env.A)
            W_0 = self.env.P(0, d[0]).permute(2, 0, 1)
            err[0] = W_0.T @ y[0] + Z.T @ y[self.env.T + 1]
            for t in range(1, self.env.T + 1):
                c_t = -self.env.r(t, d[t])
                W_t = self.env.P(t, d[t]).permute(2, 0, 1)
                err[t] = -Z.T @ y[t - 1] + W_t.T @ y[t] + z[t] - c_t
            return err.square().sum()

    def __init__(self, w: float, rho: tuple[float, float]) -> None:
        self.w = w
        self.rho = rho

    def solve(
        self,
        env: Environment,
        social_weights: list[float] | None = None,
        social_metrics: list[Callable[[torch.Tensor], float]] | None = None,
        max_iter: int = 100,
        kwargs: MESOB.Kwargs | None = None,
    ) -> torch.Tensor:
        """TODO -- what is the best way to specify social metrics and weights?"""
        social_weights = social_weights or []
        social_metrics = social_metrics or []
        kwargs = kwargs or {}

        # Initialize according to kwargs, falling back to default initialization
        d = kwargs.pop("d0", torch.ones([env.T + 1, env.S, env.A]) / env.A)
        y = kwargs.pop("y0", torch.zeros([env.T + 1, env.S]))
        z = kwargs.pop("z0", torch.zeros([env.T + 1, env.S, env.A]))

        d.requires_grad_(True)
        y.requires_grad_(True)
        z.requires_grad_(True)

        obj = self.Objective(env, social_weights, social_metrics, self.w, self.rho)

        opt_cls = kwargs.pop("opt_cls", torch.optim.Adam)
        opt_kwargs = kwargs.pop("opt_kwargs", {})
        opt = opt_cls([d, y, z], **opt_kwargs)

        y_radius = env.S * (env.T + 1) * (env.T + 2) * env.r_max / 2
        z_radius = env.S * env.A * (env.T * env.T + env.T + 2) * env.r_max

        solns = torch.empty([max_iter, env.T + 1, env.S, env.A])

        for i in range(max_iter):
            opt.zero_grad()
            obj_val = obj(d, y, z)
            obj_val.backward()
            opt.step()

            with torch.no_grad():
                d = project_simplex(d, axis=0)
                y = project_l2_ball(y, r=y_radius)
                z = project_Z_set(z, r=z_radius)

                solns[i] = d.clone()

        return solns


@torch.no_grad()
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
        rho = torch.sum(mu * j - cumsum + r > 0.0, dim=1)
        theta = (cumsum[:, rho - 1] - r) / rho
        return torch.clamp(v - theta, min=0.0)

    shape = v.shape

    if len(shape) == 1:
        v_2d = torch.unsqueeze(v, 0)
        return project_simplex_2d(v_2d)[0]
    else:
        axis = axis % len(shape)
        v_2d = v.movedim(axis, 0).flatten(start_dim=1)
        w_2d = project_simplex_2d(v_2d)
        w_2d = w_2d.reshape([shape[axis], *shape[1:axis], shape[0], *shape[axis:]])
        return w_2d.movedim(axis, 0)


@torch.no_grad()
def project_l2_ball(v: torch.Tensor, *, r: float = 1.0) -> torch.Tensor:
    return v * min(1, 1 / v.norm(p="fro")) * r


@torch.no_grad()
def project_Z_set(z: torch.Tensor, *, r: float) -> torch.Tensor:
    """TODO -- is this supposed to be a real projection?"""
    if (z >= 0).all() and z.sum() <= r:
        return z
    else:
        z_ravel = z.ravel()
        p_ind = torch.argwhere(z_ravel < 0)
        np_ind = torch.argwhere(z_ravel >= 0)
        z_ravel[p_ind] = project_simplex(z_ravel[p_ind], r=r)
        z_ravel[np_ind] = 0.0
        return z_ravel.reshape(z.shape)
