from __future__ import annotations

import sys
from dataclasses import dataclass

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import optuna
import torch

from mfglib.alg.abc import Iterative
from mfglib.alg.greedy_policy_given_mean_field import Greedy_Policy
from mfglib.env import Environment
from mfglib.mean_field import mean_field


@dataclass
class State:
    i: int
    env: Environment
    pi: torch.Tensor


class FictitiousPlay(Iterative[State]):
    """Fictitious Play algorithm.

    Notes
    -----
    The implementation is based on Fictitious Play Damped.

    When ``alpha=0.0``, the algorithm is the same as the original Fictitious Play
    algorithm. When ``alpha=1``, the algorithm is the same as Fixed Point Iteration
    algorithm.

    .. seealso::

        Refer to :cite:t:`perrin2020` and :cite:t:`perolat2021` for details.
    """

    def __init__(self, alpha: float = 0.0) -> None:
        """Fictitious Play algorithm.

        Attributes
        ----------
        alpha
            Learning rate hyperparameter. If set to 0.0, in iteration n the
            learning rate is 1 / (n + 1).
        """
        if not isinstance(alpha, (int, float)) or not 0 <= alpha <= 1:
            raise ValueError("parameter 'alpha' must be a float in [0, 1]")
        self.alpha = alpha

    def __str__(self) -> str:
        """Represent algorithm instance and associated parameters with a string."""
        return f"FictitiousPlay(alpha={self.alpha})"

    def init_state(self, env: Environment, pi_0: torch.Tensor) -> State:
        return State(i=0, env=env, pi=pi_0)

    def step_next_state(self, state: State) -> State:
        L = mean_field(state.env, state.pi)
        pi_br = Greedy_Policy(state.env, L)
        L_br = mean_field(state.env, pi_br)

        pi = state.pi
        states_dim = len(state.env.S)

        mu = L.flatten(start_dim=1 + states_dim).sum(dim=-1, keepdim=True)
        mu_br = L_br.flatten(start_dim=1 + states_dim).sum(dim=-1, keepdim=True)

        alpha_i = self.alpha if self.alpha else 1 / (state.i + 2)

        numer = (1 - alpha_i) * mu * pi + alpha_i * mu_br * pi_br
        denom = (1 - alpha_i) * mu + alpha_i * mu_br
        pi = (numer / denom).nan_to_num(nan=1 / state.env.n_actions)

        return State(i=state.i + 1, env=state.env, pi=pi)

    @property
    def parameters(self) -> dict[str, float | str | None]:
        return {"alpha": self.alpha}

    def _init_tuner_instance(self: Self, trial: optuna.Trial) -> Self:
        return type(self)(alpha=trial.suggest_float("alpha", 0.0, 1.0))
