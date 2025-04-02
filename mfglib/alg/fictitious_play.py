from __future__ import annotations

import time
from typing import Literal

import optuna
import torch

from mfglib.alg.abc import DEFAULT_ATOL, DEFAULT_MAX_ITER, DEFAULT_RTOL, Algorithm
from mfglib.alg.greedy_policy_given_mean_field import Greedy_Policy
from mfglib.alg.utils import (
    Printer,
    _ensure_free_tensor,
    _trigger_early_stopping,
    tuple_prod,
)
from mfglib.env import Environment
from mfglib.mean_field import mean_field
from mfglib.scoring import exploitability_score


class FictitiousPlay(Algorithm):
    """The **Fictitious Play** algorithm.

    The implementation is based on **Fictitious Play Damped**. The damped
    version generalizes the original algorithm by adding a learning rate
    parameter ``alpha``.

    When ``alpha=None``, the algorithm is the same as the original
    **Fictitious Play** algorithm. When ``alpha=1``, the algorithm is the same
    as the  **Fixed Point Iteration** algorithm.

    Parameters
    ----------
    alpha
        Learning rate hyperparameter. If ``None``, in iteration ``n`` the
        learning rate is set to ``1 / (n + 1)``.

    References
    ----------
    .. [#] Perrin, Sarah, et al. "Fictitious play for mean field games: Continuous
        time analysis and applications." Advances in Neural Information Processing
        Systems 33 (2020): 13199-13213. https://arxiv.org/abs/2007.03458

    .. [#] Perolat, Julien, et al. "Scaling up mean field games with online mirror
        descent." arXiv preprint arXiv:2103.00623 (2021). https://arxiv.org/abs/2103.00623

    """

    def __init__(self, alpha: float | None = None) -> None:
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
        verbose: int = 0,
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
        scores = [exploitability_score(env_instance, pi)]
        runtimes = [0.0]

        printer = Printer.setup(
            verbose=verbose,
            env_instance=env_instance,
            solver="FictitiousPlay",
            parameters={"alpha": self.alpha},
            atol=atol,
            rtol=rtol,
            max_iter=max_iter,
            expl_0=scores[0],
        )

        if _trigger_early_stopping(scores[0], scores[0], atol, rtol):
            printer.alert_early_stopping()
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
            runtimes.append(time.time() - t)

            printer.notify_of_solution(n=n, expl_n=scores[n], runtime_n=runtimes[n])

            if _trigger_early_stopping(scores[0], scores[n], atol, rtol):
                printer.alert_early_stopping()
                return solutions, scores, runtimes

        printer.alert_iterations_exhausted()
        return solutions, scores, runtimes

    @classmethod
    def _init_tuner_instance(cls, trial: optuna.Trial) -> FictitiousPlay:
        alpha_bool = trial.suggest_categorical("alpha_bool", [False, True])
        alpha_num = trial.suggest_float("alpha_num", 0.0, 1.0)
        alpha = None if alpha_bool else alpha_num
        return FictitiousPlay(alpha=alpha)

    @classmethod
    def from_study(cls, study: optuna.Study) -> "FictitiousPlay":
        best_params = study.best_params
        alpha_bool = best_params.pop("alpha_bool")
        alpha_num = best_params.pop("alpha_num")
        alpha = None if alpha_bool else alpha_num
        return cls(alpha=alpha)
