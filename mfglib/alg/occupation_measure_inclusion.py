from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import numpy as np
import optuna
import osqp
import torch
from numpy.typing import NDArray
from scipy import sparse

from mfglib.alg.abc import Iterative
from mfglib.alg.mf_omo_params import mf_omo_params
from mfglib.env import Environment
from mfglib.utils import mean_field_from_policy, policy_from_mean_field


# TODO: consider update vectors to be more efficient
# https://osqp.org/docs/interfaces/python.html#python-interface
def osqp_proj(
    d: torch.Tensor,
    b: torch.Tensor,
    A: torch.Tensor,
    x0: NDArray[Any] | None,
    y0: NDArray[Any] | None,
    eps_abs: float,
    eps_rel: float,
) -> tuple[torch.Tensor, NDArray[Any], NDArray[Any]]:
    """Project d onto Ad=b, d>=0."""
    # Problem dimensions
    n = d.size(dim=0)

    # Define the P matrix (2 * I)
    P = 2 * sparse.eye(n, format="csc")

    # Define the q vector (-2 * a)
    q: NDArray[Any] = -2 * d.numpy()

    # Define the constraints l and u
    l = np.concatenate([b.numpy(), np.zeros(n)])
    u = np.concatenate([b.numpy(), np.full(n, np.inf)])

    # Define the constraint matrix
    A_constraint = sparse.vstack([A.numpy(), sparse.eye(n, format="csc")], format="csc")

    prob = osqp.OSQP()
    prob.setup(
        P, q, A_constraint, l, u, verbose=False, eps_abs=eps_abs, eps_rel=eps_rel
    )
    if x0 is not None and y0 is not None:
        prob.warm_start(x=x0, y=y0)
    res = prob.solve()

    # numpy default is double which is fine; but to get matmul(A, sol) work needs
    # both to be same type
    sol = torch.tensor(res.x).float()

    return sol, res.x, res.y


@dataclass
class State:
    env: Environment
    pi: torch.Tensor
    d: torch.Tensor
    x0: torch.Tensor | None
    y0: torch.Tensor | None


class OccupationMeasureInclusion(Iterative[State]):
    """Mean-Field Occupation Measure Inclusion with Forward-Backward Splitting.

    Notes
    -----
    MF-OMI-FBS recasts the objective of finding a mean-field Nash equilibrium
    as an inclusion problem with occupation-measure variables. The algorithm
    is known to have polynomial regret bounds in games with the Lasry-Lions
    monotonicity property.

    .. seealso::

        Refer to :cite:t:`hu2024` for additional details.
    """

    def __init__(
        self,
        alpha: float = 1e-3,
        eta: float = 0.0,
        osqp_atol: float | None = None,
        osqp_rtol: float | None = None,
        osqp_warmstart: bool = True,
    ) -> None:
        """

        Attributes
        ----------
        alpha
            Strictly positive stepsize.
        eta
            Non-negative perturbation coefficient. Increasing eta can accelerate convergence at
            the cost of asymptotic suboptimality.
        osqp_atol
            Absolute tolerance criteria for early stopping of OSQP projection inner steps.
        osqp_rtol
            Relative tolerance criteria for early stopping of OSQP projection inner steps.
        osqp_warmstart
            Reuse previously found projections in osqp.
        """
        self.alpha = alpha
        self.eta = eta
        self.osqp_atol = osqp_atol
        self.osqp_rtol = osqp_rtol
        self.osqp_warmstart = osqp_warmstart

    def __str__(self) -> str:
        """Represent algorithm instance and associated parameters with a string."""
        return f"OccupationMeasureInclusion({self.alpha=}, {self.eta=})"

    def init_state(self, env: Environment, pi_0: torch.Tensor) -> State:
        d = mean_field_from_policy(pi_0, env=env)
        return State(env=env, pi=pi_0, d=d, x0=None, y0=None)

    def step_next_state(
        self, state: State, atol: float | None, rtol: float | None
    ) -> State:
        d = state.d
        x0, y0 = state.x0, state.y0
        d_shape = list(d.shape)
        b, A_d, c_d = mf_omo_params(state.env, d)
        d -= self.alpha * (c_d.reshape(*d_shape) + self.eta * d)

        if self.osqp_atol is None:
            osqp_atol = 1e-8 if atol is None else atol
        else:
            osqp_atol = self.osqp_atol

        if self.osqp_rtol is None:
            osqp_rtol = 1e-8 if rtol is None else rtol
        else:
            osqp_rtol = self.osqp_rtol

        # NOTE: The unused-ignore tag can be removed when we drop support for Python 3.9
        d, x0, y0 = osqp_proj(d.flatten(), b, A_d, x0, y0, osqp_atol, osqp_rtol)  # type: ignore[assignment,arg-type,unused-ignore]
        d = d.reshape(*d_shape)
        pi = policy_from_mean_field(
            d.clone().detach(), env=state.env, tol=(osqp_atol + osqp_rtol) / 2 * 10
        )

        if self.osqp_warmstart:
            return State(env=state.env, pi=pi, d=d, x0=x0, y0=y0)
        else:
            return State(env=state.env, pi=pi, d=d, x0=None, y0=None)

    @property
    def parameters(self) -> dict[str, float | str | None]:
        return {"alpha": self.alpha, "eta": self.eta}

    def _init_tuner_instance(self: Self, trial: optuna.Trial) -> Self:
        return type(self)(
            alpha=trial.suggest_float("alpha", 1e-10, 1e3, log=True),
            eta=self.eta,
            osqp_atol=self.osqp_atol,
            osqp_rtol=self.osqp_rtol,
            osqp_warmstart=self.osqp_warmstart,
        )

    def from_study(self: Self, study: optuna.Study) -> Self:
        err_msg = f"{study.best_params.keys()=} but should only contain 'alpha'."
        assert study.best_params.keys() == {"alpha"}, err_msg

        return type(self)(
            **study.best_params,
            eta=self.eta,
            osqp_atol=self.osqp_atol,
            osqp_rtol=self.osqp_rtol,
            osqp_warmstart=self.osqp_warmstart,
        )
