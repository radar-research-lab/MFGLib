from __future__ import annotations

import time
from typing import Literal

import optuna
import torch

from mfglib.alg.abc import DEFAULT_ATOL, DEFAULT_MAX_ITER, DEFAULT_RTOL, Algorithm
from mfglib.alg.q_fn import QFn
from mfglib.alg.utils import Printer, _ensure_free_tensor, _trigger_early_stopping
from mfglib.env import Environment
from mfglib.mean_field import mean_field
from mfglib.scoring import exploitability_score


class PriorDescent(Algorithm):
    """Prior Descent algorithm.

    Notes
    -----
    When `n_inner=None`, the algorithm is the same as GMF-V.

    See [#pd1]_ and [#pd2]_ for algorithm details.

    .. [#pd1] Cui, Kai, and Heinz Koeppl. "Approximately solving mean field games via
        entropy-regularized deep reinforcement learning." International Conference
        on Artificial Intelligence and Statistics. PMLR, 2021.
        https://proceedings.mlr.press/v130/cui21a.html

    .. [#pd2] Guo, Xin, et al. "Learning mean-field games." Advances in Neural
        Information Processing Systems 32 (2019). https://arxiv.org/abs/1901.09585
    """

    def __init__(self, eta: float = 1.0, n_inner: int | None = None) -> None:
        """Prior Descent algorithm.

        Attributes
        ----------
        eta
            Temperature.
        n_inner
            The prior is updated every `n_inner` iterations. If None,
            the prior remains intact.
        """
        if not isinstance(eta, (int, float)) or eta <= 0:
            raise ValueError("`eta` must be a positive float")
        if n_inner:
            if not isinstance(n_inner, int) or n_inner <= 0:
                raise ValueError("if not None, `n_inner` must be a positive integer")
        self.eta = eta
        self.n_inner = n_inner

    def __str__(self) -> str:
        """Represent algorithm instance and associated parameters with a string."""
        return f"PriorDescent(eta={self.eta}, n_inner={self.n_inner})"

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
            A numpy array of size (T+1,)+S+A representing the initial policy. If
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

        l_s = len(S)
        l_a = len(A)
        ones_ts = (1,) * (1 + l_s)
        ats_to_tsa = tuple(range(l_a, l_a + 1 + l_s)) + tuple(range(l_a))

        pi = _ensure_free_tensor(pi, env_instance)
        q = pi.clone()

        solutions = [pi]
        argmin = 0
        scores = [exploitability_score(env_instance, pi)]
        runtimes = [0.0]

        printer = Printer(verbose=verbose)
        printer.print_info_panels(
            env_instance=env_instance,
            cls="PriorDescent",
            parameters={"eta": self.eta, "n_inner": self.n_inner},
            atol=atol,
            rtol=rtol,
            max_iter=max_iter,
        )
        printer.add_table_row(
            n=0,
            expl_n=scores[0],
            expl_0=scores[0],
            argmin=argmin,
            runtime_n=runtimes[0],
        )

        if _trigger_early_stopping(scores[0], scores[0], atol, rtol):
            printer.alert_stopped_early()
            return solutions, scores, runtimes

        t = time.time()
        for n in range(1, max_iter + 1):
            # Mean-field corresponding to the policy
            L = mean_field(env_instance, pi)

            # Q-function corresponding to the mean-field
            Q = QFn(env_instance, L, verify_integrity=False).optimal() / self.eta

            # Compute the next policy
            Q_exp = torch.exp(Q)
            q_Q_exp = q.mul(Q_exp)
            q_Q_exp_sum_rptd = (
                q_Q_exp.flatten(start_dim=1 + l_s)
                .sum(-1)
                .repeat(A + ones_ts)
                .permute(ats_to_tsa)
            )
            pi = q_Q_exp.div(q_Q_exp_sum_rptd)

            # Update the prior
            if self.n_inner and (n + 1) % self.n_inner == 0:
                q = pi.clone()

            solutions.append(pi.clone().detach())
            scores.append(exploitability_score(env_instance, pi))
            if scores[n] < scores[argmin]:
                argmin = n
            runtimes.append(time.time() - t)

            printer.add_table_row(
                n=n,
                expl_n=scores[n],
                expl_0=scores[0],
                argmin=argmin,
                runtime_n=runtimes[n],
            )

            if _trigger_early_stopping(scores[0], scores[n], atol, rtol):
                printer.alert_stopped_early()
                return solutions, scores, runtimes

        printer.alert_iterations_exhausted()

        return solutions, scores, runtimes

    @classmethod
    def _init_tuner_instance(cls, trial: optuna.Trial) -> PriorDescent:
        n_inner_bool = trial.suggest_categorical("n_inner_bool", [False, True])
        n_inner_num = trial.suggest_int("n_inner_num", 1, 101, step=5)
        n_inner = None if n_inner_bool else n_inner_num
        return PriorDescent(
            eta=trial.suggest_float("eta", 1e-5, 1e5, log=True),
            n_inner=n_inner,
        )
