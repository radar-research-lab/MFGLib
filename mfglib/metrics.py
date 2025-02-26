from __future__ import annotations

from typing import cast, overload

import torch

from mfglib.env import Environment
from mfglib.mean_field import mean_field
from mfglib.q_fn import QFn

__all__ = ["exploitability_score"]


@overload
def exploitability_score(env_instance: Environment, pi: torch.Tensor) -> float: ...


@overload
def exploitability_score(
    env_instance: Environment, pi: list[torch.Tensor]
) -> list[float]: ...


def exploitability_score(
    env_instance: Environment, pi: torch.Tensor | list[torch.Tensor]
) -> float | list[float]:
    """Compute the exploitability metric for a given policy (or policies)."""
    if isinstance(pi, list):
        return [exploitability_score(env_instance, x) for x in pi]

    mu0 = env_instance.mu0
    l_S = len(env_instance.S)

    gamma_pi = mean_field(env_instance, pi)

    q_fn = QFn(env_instance, gamma_pi, verify_integrity=False)

    Q_s_gamma_pi = q_fn.optimal()
    Q_gamma_pi_pi = q_fn.for_policy(pi)

    max_Q_s_gamma_pi_0, _ = Q_s_gamma_pi[0].flatten(start_dim=l_S).max(dim=-1)
    expl1 = (mu0 * max_Q_s_gamma_pi_0).sum()
    pi_Q_gamma_pi_pi_0 = (pi[0] * Q_gamma_pi_pi[0]).flatten(start_dim=l_S).sum(dim=-1)
    expl2 = (mu0 * pi_Q_gamma_pi_pi_0).sum()
    expl = expl1 - expl2

    return cast(float, expl.item())
