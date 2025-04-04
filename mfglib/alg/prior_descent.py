from __future__ import annotations

from dataclasses import dataclass

import optuna
import torch

from mfglib.alg.abc import Iterative
from mfglib.alg.q_fn import QFn
from mfglib.env import Environment
from mfglib.mean_field import mean_field


@dataclass
class State:
    i: int
    env: Environment
    pi: torch.Tensor
    q: torch.Tensor


class PriorDescent(Iterative[State]):
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

    def init_state(self, env: Environment, pi_0: torch.Tensor) -> State:
        return State(i=0, env=env, pi=pi_0, q=pi_0.clone())

    def step_next_state(self, state: State) -> State:
        L = mean_field(state.env, state.pi)
        Q = QFn(state.env, L, verify_integrity=False).optimal() / self.eta

        l_s = len(state.env.S)
        l_a = len(state.env.A)
        ones_ts = (1,) * (1 + l_s)
        ats_to_tsa = tuple(range(l_a, l_a + 1 + l_s)) + tuple(range(l_a))
        Q_exp = torch.exp(Q)
        q_Q_exp = state.q.mul(Q_exp)
        q_Q_exp_sum_rptd = (
            q_Q_exp.flatten(start_dim=1 + l_s)
            .sum(-1)
            .repeat(state.env.A + ones_ts)
            .permute(ats_to_tsa)
        )
        pi = q_Q_exp.div(q_Q_exp_sum_rptd)

        if self.n_inner and (state.i + 2) % self.n_inner == 0:
            q = pi.clone()
        else:
            q = state.q
        return State(i=state.i + 1, env=state.env, pi=pi, q=q)

    @property
    def parameters(self) -> dict[str, float | str | None]:
        return {"eta": self.eta, "n_inner": self.n_inner}

    @classmethod
    def _init_tuner_instance(cls, trial: optuna.Trial) -> PriorDescent:
        n_inner_bool = trial.suggest_categorical("n_inner_bool", [False, True])
        n_inner_num = trial.suggest_int("n_inner_num", 1, 101, step=5)
        n_inner = None if n_inner_bool else n_inner_num
        return PriorDescent(
            eta=trial.suggest_float("eta", 1e-5, 1e5, log=True),
            n_inner=n_inner,
        )
