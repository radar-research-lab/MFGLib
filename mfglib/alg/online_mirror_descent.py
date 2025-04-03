from __future__ import annotations

import dataclasses

import optuna
import torch

import mfglib
from mfglib.alg.abc import Algorithm
from mfglib.alg.q_fn import QFn
from mfglib.env import Environment
from mfglib.mean_field import mean_field


@dataclasses.dataclass
class State(mfglib.alg.abc.State):
    def __init__(self, env: Environment, pi_0: torch.Tensor) -> None:
        super().__init__(pi_i=pi_0)
        self.env = env
        self.y = torch.zeros(env.T + 1, *env.S, *env.A)

    def next(self, pi_i: torch.Tensor) -> State:
        self.pi_i = pi_i
        return self


class OnlineMirrorDescent(Algorithm[State]):
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

    def init_state(self, env: Environment, pi_0: torch.Tensor) -> State:
        return State(env=env, pi_0=pi_0)

    def step_state(self, state: State) -> State:
        L = mean_field(state.env, state.pi_i)
        Q = QFn(state.env, L, verify_integrity=False).for_policy(state.pi_i)
        state.y += self.alpha * Q
        states_dim = len(state.env.S)
        y_flat = state.y.flatten(start_dim=1 + states_dim)
        softmax = torch.nn.Softmax(dim=-1)
        pi_i = softmax(y_flat).reshape(state.env.T + 1, *state.env.S, *state.env.A)
        return state.next(pi_i)

    @property
    def parameters(self) -> dict[str, float | str | None]:
        return {"alpha": self.alpha}

    @classmethod
    def _init_tuner_instance(cls, trial: optuna.Trial) -> OnlineMirrorDescent:
        return OnlineMirrorDescent(
            alpha=trial.suggest_float("alpha", 1e-5, 1e5, log=True),
        )
