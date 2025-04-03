from __future__ import annotations

import dataclasses

import optuna
import torch

import mfglib.alg.abc
from mfglib.alg.abc import Algorithm
from mfglib.alg.greedy_policy_given_mean_field import Greedy_Policy
from mfglib.env import Environment
from mfglib.mean_field import mean_field


@dataclasses.dataclass
class State(mfglib.alg.abc.State):
    def __init__(self, env: Environment, pi_0: torch.Tensor) -> None:
        super().__init__(pi_i=pi_0)
        self.env = env
        self.i = 0

    def next(self, pi_i: torch.Tensor) -> State:
        self.i += 1
        self.pi_i = pi_i
        return self


class FictitiousPlay(Algorithm[State]):
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

    def init_state(self, env: Environment, pi_0: torch.Tensor) -> State:
        return State(env=env, pi_0=pi_0)

    def step_state(self, state: State) -> State:
        L = mean_field(state.env, state.pi_i)
        pi_br = Greedy_Policy(state.env, L)
        L_br = mean_field(state.env, pi_br)

        pi_i = state.pi_i
        states_dim = len(state.env.S)

        mu = L.flatten(start_dim=1 + states_dim).sum(dim=-1, keepdim=True)
        mu_br = L_br.flatten(start_dim=1 + states_dim).sum(dim=-1, keepdim=True)

        alpha_i = self.alpha if self.alpha else 1 / (state.i + 1)

        numer = (1 - alpha_i) * mu * pi_i + alpha_i * mu_br * pi_br
        denom = (1 - alpha_i) * mu + alpha_i * mu_br
        pi_i = (numer / denom).nan_to_num(nan=1 / state.env.n_actions)

        return state.next(pi_i)

    @property
    def parameters(self) -> dict[str, float | str | None]:
        return {"alpha": self.alpha}

    @classmethod
    def _init_tuner_instance(cls, trial: optuna.Trial) -> FictitiousPlay:
        alpha_bool = trial.suggest_categorical("alpha_bool", [False, True])
        alpha_num = trial.suggest_float("alpha_num", 0.0, 1.0)
        alpha = None if alpha_bool else alpha_num
        return FictitiousPlay(alpha=alpha)
