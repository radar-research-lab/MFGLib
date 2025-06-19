from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import optuna
import torch

from mfglib.alg.abc import Iterative
from mfglib.alg.q_fn import QFn
from mfglib.env import Environment
from mfglib.mean_field import mean_field


@dataclass
class State:
    env: Environment
    pi: torch.Tensor
    y: torch.Tensor


class OnlineMirrorDescent(Iterative[State]):
    """Online Mirror Descent algorithm.

    .. seealso::

        Refer to :cite:t:`perolat2021` for additional details.
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

    def init_state(
        self,
        env: Environment,
        pi_0: torch.Tensor,
        atol: float | None,
        rtol: float | None,
    ) -> State:
        return State(env=env, pi=pi_0, y=torch.zeros(env.T + 1, *env.S, *env.A))

    def step_next_state(self, state: State) -> State:
        L = mean_field(state.env, state.pi)
        Q = QFn(state.env, L, verify_integrity=False).for_policy(state.pi)
        y = state.y + self.alpha * Q
        n_state_coords = len(state.env.S)
        y_flat = y.flatten(start_dim=1 + n_state_coords)
        softmax = torch.nn.Softmax(dim=-1)
        pi = softmax(y_flat).reshape(state.env.T + 1, *state.env.S, *state.env.A)
        return State(env=state.env, pi=pi, y=y)

    @property
    def parameters(self) -> dict[str, float | str | None]:
        return {"alpha": self.alpha}

    def _init_tuner_instance(self: Self, trial: optuna.Trial) -> Self:
        return type(self)(alpha=trial.suggest_float("alpha", 1e-5, 1e5, log=True))
