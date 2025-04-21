from __future__ import annotations

import time
from typing import Literal

import numpy as np
import optuna
import osqp
import torch
from scipy import sparse

from mfglib.alg.abc import DEFAULT_ATOL, DEFAULT_MAX_ITER, DEFAULT_RTOL, Algorithm
from mfglib.alg.mf_omo_params import mf_omo_params
from mfglib.alg.utils import (
    _ensure_free_tensor,
    _print_fancy_header,
    _print_fancy_table_row,
    _print_solve_complete,
    _trigger_early_stopping,
    extract_policy_from_mean_field,
)
from mfglib.env import Environment
from mfglib.mean_field import mean_field
from mfglib.scoring import exploitability_score


# TODO: Change to support warm start and update vectors to be more efficient
#  https://osqp.org/docs/interfaces/python.html#python-interface
def osqp_proj(d: torch.Tensor, b: torch.Tensor, A: torch.Tensor, eps_abs: float=1e-8, eps_rel: float=1e-8) -> torch.Tensor:
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
    prob.setup(P, q, A_constraint, l, u, verbose=False, eps_abs=eps_abs, eps_rel=eps_rel)
    res = prob.solve()

    # numpy default is double which is fine; but to get matmul(A, sol) work needs
    # both to be same type
    sol = torch.tensor(res.x).float()

    return sol


class OccupationMeasureInclusion(Algorithm):
    """Mean-Field Occupation Measure Inclusion with Forward-Backward Splitting.

    Notes
    -----
    MF-OMI-FBS recasts the objective of finding a mean-field Nash equilibrium
    as an inclusion problem with occupation-measure variables. The algorithm
    is known to have polynomial regret bounds in games with the Lasry-Lions
    monotonicity property.
    """

    def __init__(self, alpha: float = 1.0, eta: float = 0.0) -> None:
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

    def solve(
        self,
        env_instance: Environment,
        *,
        pi: Literal["uniform"] | torch.Tensor = "uniform",
        max_iter: int = DEFAULT_MAX_ITER,
        atol: float | None = DEFAULT_ATOL,
        rtol: float | None = DEFAULT_RTOL,
        osqp_atol: float | None = None,
        osqp_rtol: float | None = None,
        verbose: bool = False,
    ) -> tuple[list[torch.Tensor], list[float], list[float]]:
        """Run the algorithm and solve for a Nash-Equilibrium policy.

        Args
        ----
        env_instance
            An instance of a specific environment.
        pi
            A numpy array of size (T+1,)+S+A representing the initial policy.
            If 'uniform', the initial policy will be the uniform distribution.
        max_iter
            Maximum number of iterations to run.
        atol
            Absolute tolerance criteria for early stopping.
        rtol
            Relative tolerance criteria for early stopping.
        osqp_atol
            Absolute tolerance criteria for early stopping of OSPQ projection inner steps.
        osqp_rtol
            Relative tolerance criteria for early stopping of OSPQ projection inner steps.
        verbose
            Print convergence information during iteration.
        """
        osqp_atol = atol if osqp_atol is None else osqp_atol
        osqp_rtol = rtol if osqp_rtol is None else osqp_rtol

        pi = _ensure_free_tensor(pi, env_instance)

        solutions = [pi]
        argmin = 0
        scores = [exploitability_score(env_instance, pi)]
        runtimes = [0.0]

        if verbose:
            _print_fancy_header(
                alg_instance=self,
                env_instance=env_instance,
                max_iter=max_iter,
                atol=atol,
                rtol=rtol,
            )
            _print_fancy_table_row(
                n=0,
                score_n=scores[0],
                score_0=scores[0],
                argmin=argmin,
                runtime_n=runtimes[0],
            )

        if _trigger_early_stopping(scores[0], scores[0], atol, rtol):
            if verbose:
                _print_solve_complete(seconds_elapsed=runtimes[0])
            return solutions, scores, runtimes

        # initialize d = L^pi
        d = mean_field(env_instance, pi)
        d_shape = list(d.shape)  # non-flattened shape of d

        t = time.time()
        for n in range(1, max_iter + 1):
            # obtain occupation-measure params
            b, A_d, c_d = mf_omo_params(env_instance, d)

            # Update d and pi
            d -= self.alpha * (c_d.reshape(*d_shape) + self.eta * d)
            d = osqp_proj(d.flatten(), b, A_d, osqp_atol, osqp_rtol).reshape(*d_shape)
            pi = extract_policy_from_mean_field(env_instance, d.clone().detach())

            solutions.append(
                pi.clone().detach()
            )  # do we need to clone + detach again? no? same for MF-OMO？
            scores.append(exploitability_score(env_instance, pi))
            if scores[n] < scores[argmin]:
                argmin = n
            runtimes.append(time.time() - t)

            if verbose:
                _print_fancy_table_row(
                    n=n,
                    score_n=scores[n],
                    score_0=scores[0],
                    argmin=argmin,
                    runtime_n=runtimes[n],
                )

            if _trigger_early_stopping(scores[0], scores[n], atol, rtol):
                if verbose:
                    _print_solve_complete(seconds_elapsed=runtimes[n])
                return solutions, scores, runtimes

        if verbose:
            _print_solve_complete(seconds_elapsed=time.time() - t)

        return solutions, scores, runtimes

    @classmethod
    def _init_tuner_instance(cls, trial: optuna.Trial) -> OccupationMeasureInclusion:
        return OccupationMeasureInclusion(
            alpha=trial.suggest_float("alpha", 1e-10, 1e3, log=True),
        )
