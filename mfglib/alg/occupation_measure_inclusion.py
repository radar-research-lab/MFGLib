from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import optuna
import osqp
import torch
from scipy import sparse

from mfglib.alg.abc import Iterative
from mfglib.alg.mf_omo_params import mf_omo_params
from mfglib.alg.utils import extract_policy_from_mean_field
from mfglib.env import Environment
from mfglib.mean_field import mean_field


# TODO: Change to support warm start and update vectors to be more efficient
#  https://osqp.org/docs/interfaces/python.html#python-interface
def osqp_proj(d: torch.Tensor, b: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Project d onto Ad=b, d>=0."""
    # Problem dimensions
    n = d.size(dim=0)

    # Define the P matrix (2 * I)
    P = 2 * sparse.eye(n, format="csc")

    # Define the q vector (-2 * a)
    q = -2 * d.numpy()

    # Define the constraints l and u
    l = np.concatenate([b.numpy(), np.zeros(n)])
    u = np.concatenate([b.numpy(), np.full(n, np.inf)])

    # Define the constraint matrix
    A_constraint = sparse.vstack([A.numpy(), sparse.eye(n, format="csc")], format="csc")

    prob = osqp.OSQP()
    prob.setup(P, q, A_constraint, l, u, verbose=False, eps_abs=1e-8, eps_rel=1e-8)
    res = prob.solve()

    # numpy default is double which is fine; but to get matmul(A, sol) work needs
    # both to be same type
    sol = torch.tensor(res.x).float()

    return sol


@dataclass
class State:
    env: Environment
    pi: torch.Tensor
    d: torch.Tensor


class OccupationMeasureInclusion(Iterative[State]):
    """Mean-Field Occupation Measure Inclusion with Forward-Backward Splitting.

    Notes
    -----
    MF-OMI-FBS recasts the objective of finding a mean-field Nash equilibrium
    as an inclusion problem with occupation-measure variables. The algorithm
    is known to have polynomial regret bounds in games with the Lasry-Lions
    monotonicity property.
    """

    def __init__(self, alpha: float = 0.001, eta: float = 0.0) -> None:
        """

        Attributes
        ----------
        alpha
            Strictly positive stepsize.
        eta
            Non-negative perturbation coefficient. Increasing eta can accelerate convergence at
            the cost of asymptotic suboptimality.
        """
        self.alpha = alpha
        self.eta = eta

    def __str__(self) -> str:
        """Represent algorithm instance and associated parameters with a string."""
        return f"OccupationMeasureInclusion({self.alpha=}, {self.eta=})"

    def init_state(self, env: Environment, pi_0: torch.Tensor) -> State:
        d = mean_field(env, pi_0)
        return State(env=env, pi=pi_0, d=d)

    def step_next_state(self, state: State) -> State:
        d = state.d
        d_shape = list(d.shape)
        b, A_d, c_d = mf_omo_params(state.env, d)
        d -= self.alpha * (c_d.reshape(*d_shape) + self.eta * d)
        d = osqp_proj(d.flatten(), b, A_d).reshape(*d_shape)
        pi = extract_policy_from_mean_field(state.env, d.clone().detach())
        return State(env=state.env, pi=pi, d=d)

    @property
    def parameters(self) -> dict[str, float | str | None]:
        return {"alpha": self.alpha, "eta": self.eta}

    @classmethod
    def _init_tuner_instance(cls, trial: optuna.Trial) -> OccupationMeasureInclusion:
        return OccupationMeasureInclusion(
            alpha=trial.suggest_float("alpha", 1e-10, 1e3, log=True),
        )
