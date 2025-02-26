from __future__ import annotations

import time
from typing import Literal

import optuna
import torch

from mfglib.alg.abc import DEFAULT_ATOL, DEFAULT_MAX_ITER, DEFAULT_RTOL, Algorithm
from mfglib.alg.greedy_policy_given_mean_field import Greedy_Policy
from mfglib.alg.utils import (
    _ensure_free_tensor,
    _print_fancy_header,
    _print_fancy_table_row,
    _print_solve_complete,
    _trigger_early_stopping,
    tuple_prod,
)
from mfglib.env import Environment
from mfglib.mean_field import mean_field
from mfglib.scoring import exploitability_score


class FictitiousPlay(Algorithm):
    """Fictitious Play algorithm.

    Notes
    -----
    The implementation is based on Fictitious Play Damped.

    When ``alpha=None``, the algorithm is the same as the original Fictitious Play
    algorithm. When ``alpha=1``, the algorithm is the same as Fixed Point Iteration
    algorithm.

    See [#fp1]_ and [#fp2]_ for algorithm details.

    .. [#fp1] Perrin, Sarah, et al. "Fictitious play for mean field games: Continuous
        time analysis and applications." Advances in Neural Information Processing
        Systems 33 (2020): 13199-13213. https://arxiv.org/abs/2007.03458

    .. [#fp2] Perolat, Julien, et al. "Scaling up mean field games with online mirror
        descent." arXiv preprint arXiv:2103.00623 (2021).
        https://arxiv.org/abs/2103.00623

    """

    def __init__(self, alpha: float | None = None) -> None:
        """Fictitious Play algorithm.

        Attributes
        ----------
        alpha
            Learning rate hyperparameter. If None, in iteration n the
            learning rate is 1 / (n + 1).
        """
        if alpha:
            if not isinstance(alpha, (int, float)) or not 0 <= alpha <= 1:
                raise ValueError("if not None, `alpha` must be a float in [0, 1]")
        self.alpha = alpha

    def __str__(self) -> str:
        """Represent algorithm instance and associated parameters with a string."""
        return f"FictitiousPlay(alpha={self.alpha})"

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
            A tensor of size (T+1,)+S+A representing the initial policy. If
            'uniform', the initial policy will be the uniform distribution.
        max_iter
            Maximum number of iterations to run.
        atol
            Absolute tolerance criteria for early stopping.
        rtol
            Relative tolerance criteria for early stopping.
        verbose
            Print convergence information during iteration.
        """
        S = env_instance.S
        A = env_instance.A

        # Auxiliary variables
        l_s = len(S)
        l_a = len(A)
        n_a = tuple_prod(A)
        ones_ts = (1,) * (1 + l_s)
        ats_to_tsa = tuple(range(l_a, l_a + 1 + l_s)) + tuple(range(l_a))

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
            # Compute the greedy policy and its induced mean-field
            L = mean_field(env_instance, pi)
            pi_br = Greedy_Policy(env_instance, L)
            L_br = mean_field(env_instance, pi_br)

            # Update policy
            mu_rptd = (
                L.flatten(start_dim=1 + l_s)
                .sum(-1)
                .repeat(A + ones_ts)
                .permute(ats_to_tsa)
            )
            mu_br_rptd = (
                L_br.flatten(start_dim=1 + l_s)
                .sum(-1)
                .repeat(A + ones_ts)
                .permute(ats_to_tsa)
            )
            weight = self.alpha if self.alpha else 1 / (n + 1)

            pi_next_num = (1 - weight) * pi.mul(mu_rptd) + weight * pi_br.mul(
                mu_br_rptd
            )
            pi_next_den = (1 - weight) * mu_rptd + weight * mu_br_rptd
            pi = pi_next_num.div(pi_next_den).nan_to_num(
                nan=1 / n_a, posinf=1 / n_a, neginf=1 / n_a
            )  # using uniform distribution when divided by zero

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

    @classmethod
    def _init_tuner_instance(cls, trial: optuna.Trial) -> FictitiousPlay:
        alpha_bool = trial.suggest_categorical("alpha_bool", [False, True])
        alpha_num = trial.suggest_float("alpha_num", 0.0, 1.0)
        alpha = None if alpha_bool else alpha_num
        return FictitiousPlay(alpha=alpha)
