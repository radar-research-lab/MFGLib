from __future__ import annotations

import time
from typing import Literal, cast

import optuna
import torch

from mfglib.alg.abc import DEFAULT_ATOL, DEFAULT_MAX_ITER, DEFAULT_RTOL, Algorithm
from mfglib.alg.q_fn import QFn
from mfglib.alg.utils import (
    _ensure_free_tensor,
    _print_fancy_header,
    _print_fancy_table_row,
    _print_solve_complete,
    _trigger_early_stopping,
)
from mfglib.env import Environment
from mfglib.mean_field import mean_field
from mfglib.scoring import exploitability_score


class OnlineMirrorDescent(Algorithm):
    """Online Mirror Descent algorithm.

    Notes
    -----
    See [#omd1]_ for algorithm details.

    .. [#omd1] Perolat, Julien, et al. "Scaling up mean field games with online mirror
        descent." arXiv preprint arXiv:2103.00623 (2021). https://arxiv.org/abs/2103.00623
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """Online Mirror Descent algorithm.

        Attributes
        ----------
        alpha
            Learning rate hyperparameter.
        """
        self.alpha = alpha

    def __str__(self) -> str:
        """Represent algorithm instance and associated parameters with a string."""
        return f"OnlineMirrorDescent(alpha={self.alpha})"

    def solve(
        self,
        env_instance: Environment,
        *,
        pi: Literal["uniform"] | torch.Tensor = "uniform",
        max_iter: int = DEFAULT_MAX_ITER,
        atol: float | None = DEFAULT_ATOL,
        rtol: float | None = DEFAULT_RTOL,
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

        y = torch.zeros((T + 1,) + S + A)

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

        t = time.time()
        for n in range(1, max_iter + 1):
            # Mean-field corresponding to the policy
            L = mean_field(env_instance, pi)

            # Q-function corresponding to the policy and mean-field
            Q = QFn(env_instance, L, verify_integrity=False).for_policy(pi)

            # Update y and pi
            y += self.alpha * Q
            pi = cast(
                torch.Tensor,
                soft_max(y.flatten(start_dim=1 + l_s)).reshape((T + 1,) + S + A),
            )

            solutions.append(pi.clone().detach())
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

    def _init_tuner_instance(self, trial: optuna.Trial) -> OnlineMirrorDescent:
        return OnlineMirrorDescent(
            alpha=trial.suggest_float("alpha", 1e-5, 1e5, log=True),
        )

    def from_study(self, study: optuna.Study) -> OnlineMirrorDescent:
        return # placeholder for the mfomi paper revision example test

