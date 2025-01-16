from __future__ import annotations

import copy
import time
from typing import Literal

import numpy as np
import optuna
import osqp
import torch
from scipy import sparse

from mfglib.alg.abc import Algorithm
from mfglib.alg.mf_omo_params import mf_omo_params
from mfglib.alg.mf_omo_policy_given_mean_field import (  # TODO: consider renaming this as it's used for both OMI and OMO
    mf_omo_policy,
)
from mfglib.alg.utils import (
    _ensure_free_tensor,
    _print_fancy_header,
    _print_fancy_table_row,
    _print_solve_complete,
    _trigger_early_stopping,
    project_onto_simplex,
)
from mfglib.env import Environment
from mfglib.mean_field import mean_field
from mfglib.metrics import exploitability_score

# torch.set_default_tensor_type(torch.DoubleTensor)


### TODO: Change to support warm start and update vectors to be more efficient https://osqp.org/docs/interfaces/python.html#python-interface
def osqp_proj(d: torch.Tensor, b: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Project d onto Ad=b, d>=0."""
    # Problem dimensions
    n = d.size(0)
    m = A.size(0)

    # Define the P matrix (2 * I)
    P = 2 * sparse.eye(n, format="csc")

    # Define the q vector (-2 * a)
    q = -2 * d.numpy()

    # Define the constraints l and u
    l = np.concatenate([b.numpy(), np.zeros(n)])
    u = np.concatenate([b.numpy(), np.inf * np.ones(n)])

    # Define the constraint matrix
    A_constraint = sparse.vstack([A.numpy(), sparse.eye(n, format="csc")], format="csc")

    prob = osqp.OSQP()
    prob.setup(P, q, A_constraint, l, u, verbose=False, eps_abs=1e-8, eps_rel=1e-8)
    res = prob.solve()

    sol = torch.tensor(
        res.x
    ).float()  # numpy default is double which is fine; but to get matmul(A, sol) work needs both to be same type
    ### DEBUG
    # print(np.sum(np.maximum(-sol.numpy(), 0)), np.sum(sol.numpy()) - 1, np.sum(np.abs((torch.matmul(A,sol) - b).numpy())), sol.shape, A.shape, b.shape)
    # sol_numpy_reshaped = sol.numpy().reshape(3, 4, 3)
    # print(np.sum(sol_numpy_reshaped, axis=(1,2)))
    # print(A, b)

    ### project onto >= 0
    # sol = torch.tensor(np.maximum(res.x, 0))

    ### this one is definitely wrong as should project to simplex for each timestep
    # sol = project_onto_simplex(torch.tensor(res.x))

    return sol


class OccupationMeasureInclusion(Algorithm):
    """Online Mirror Descent algorithm.

    Notes
    -----
    See [#mfoml]_ for algorithm details.

    .. [#mfoml] Hu, Anran and Zhang, Junzi "MF-OML: Online Mean-Field Reinforcement Learning
        with Occupation Measures for Large Population Games."
        arXiv preprint arxiv:2405.00282 (2024). https://arxiv.org/abs/2405.00282
    """

    def __init__(self, alpha: float = 1.0, eta: float = 0.0) -> None:
        """MF-OMI-FBS algorithm.

        Attributes
        ----------
        alpha
            Learning rate hyperparameter.
        eta
            Perturbation coefficient (to accelerate convergence at the cost of asymptotic suboptimality).
        """
        self.alpha = alpha
        self.eta = eta

    def __str__(self) -> str:
        """Represent algorithm instance and associated parameters with a string."""
        return f"OccupationMeasureInclusion(alpha={self.alpha, self.eta})"

    def solve(
        self,
        env_instance: Environment,
        *,
        pi: Literal["uniform"] | torch.Tensor = "uniform",
        max_iter: int = 100,
        atol: float | None = 1e-3,
        rtol: float | None = 1e-3,
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
        verbose
            Print convergence information during iteration.
        """
        T = env_instance.T
        S = env_instance.S
        A = env_instance.A

        # d = torch.zeros((T + 1,) + S + A)

        # Auxiliary functions
        soft_max = torch.nn.Softmax(dim=-1)

        # Auxiliary variables
        l_s = len(S)

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

            # print(n, "before update", np.sum(np.abs(d.numpy())))

            # Update d and pi
            d_old = copy.deepcopy(d.numpy())

            d -= self.alpha * (c_d.reshape(*d_shape) + self.eta * d)

            # print(f"iter: {n}, residual: {np.sum(np.abs(d.numpy()-d_old))}")

            # print(n, "after update", np.sum(np.abs(d.numpy())))

            d = osqp_proj(d.flatten(), b, A_d).reshape(*d_shape)

            # print(d)

            # pi = cast(
            #     torch.Tensor,
            #     soft_max(d.flatten(start_dim=1 + l_s)).reshape((T + 1,) + S + A),
            # ) # previous bug --> btw why did I cast a tensor to tensor?
            pi = mf_omo_policy(env_instance, d.clone().detach())

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
    def _tuner_instance(cls, trial: optuna.Trial) -> OccupationMeasureInclusion:
        return OccupationMeasureInclusion(
            alpha=trial.suggest_float("alpha", 1e-10, 1e3, log=True),
            # eta=trial.suggest_float("eta", 1e-8, 10, log=True),
        )

    def tune(
        self,
        env_suite: list[Environment],
        *,
        max_iter: int = 100,
        atol: float = 1e-3,
        rtol: float = 1e-3,
        metric: Literal["shifted_geo_mean", "failure_rate"] = "shifted_geo_mean",
        n_trials: int | None = 10,
        timeout: float = 30.0,
    ) -> OccupationMeasureInclusion:
        """Tune the algorithm over a given environment suite.

        Args
        ----
        env_suite
            A list of environment instances.
        max_iter
            The number of iterations to run the algorithm on each environment
            instance.
        atol
            Absolute tolerance criteria for early stopping.
        rtol
            Relative tolerance criteria for early stopping.
        metric
            Determines which metric to be used for scoring a trial. Either
            ``shifted_geo_mean`` or ``failure_rate``.
        n_trials
            The number of trials. If this argument is not given, as many
            trials are run as possible.
        timeout
            Stop tuning after the given number of second(s) on each
            environment instance. If this argument is not given, as many trials are
            run as possible.
        """
        params = self._optimize_optuna_study(
            env_suite=env_suite,
            max_iter=max_iter,
            atol=atol,
            rtol=rtol,
            metric=metric,
            n_trials=n_trials,
            timeout=timeout,
        )
        if params:
            self.alpha = params["alpha"]
            # self.eta = params["eta"]
        return self
