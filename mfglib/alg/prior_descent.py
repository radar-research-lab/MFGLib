from __future__ import annotations

import optuna
import torch

from mfglib.alg.abc import Iterative
from mfglib.alg.q_fn import QFn
from mfglib.env import Environment
from mfglib.mean_field import mean_field


class State:
    def __init__(self, env: Environment, pi_0: torch.Tensor) -> None:
        self.env = env
        self.pi_i = pi_0
        self.i = 0
        self.q = pi_0.clone()

    def next(self, pi_i: torch.Tensor, q: torch.Tensor) -> State:
        self.i += 1
        self.pi_i = pi_i
        self.q = q
        return self


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
        return State(env=env, pi_0=pi_0)

    def step_state(self, state: State) -> State:
        L = mean_field(state.env, state.pi_i)
        Q = QFn(state.env, L, verify_integrity=False).optimal() / self.eta
        vals = (state.q * Q.exp()).flatten(start_dim=1 + len(state.env.S))
        pi_i = torch.softmax(vals, dim=-1).reshape(
            state.env.T + 1, *state.env.S, *state.env.A
        )
        if self.n_inner and (state.i + 1) % self.n_inner == 0:
            q = pi_i.clone()
        else:
            q = state.q
        return state.next(pi_i=pi_i, q=q)

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
