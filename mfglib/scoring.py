from __future__ import annotations

from typing import cast, overload

import torch

from mfglib.alg.q_fn import QFn
from mfglib.env import Environment
from mfglib.mean_field import mean_field
from mfglib.alg.utils import tuple_prod

__all__ = ["exploitability_score"]


def postprocess_policy(env_instance: Environment, pi: torch.Tensor) -> torch.Tensor:
    """encure policy to be in the probability simplex"""
    zeros = torch.zeros(*pi.shape)
    pi = torch.maximum(pi, zeros)
    S = env_instance.S
    A = env_instance.A
    l_s = len(S)
    l_a = len(A)
    n_a = tuple_prod(A)
    ones_ts = (1,) * (l_s + 1)
    ats_to_tsa = tuple(range(l_a, l_a + 1 + l_s)) + tuple(range(l_a))
    pi_sum_rptd = (
        pi.flatten(start_dim=1 + l_s).sum(-1).repeat(A + ones_ts).permute(ats_to_tsa)
    )
    pi = pi.div(pi_sum_rptd).nan_to_num(
        nan=1 / n_a, posinf=1 / n_a, neginf=1 / n_a
    )  # using uniform distribution when L_t_sum_rptd is zero

    return pi


@overload
def exploitability_score(
    env_instance: Environment, pi: torch.Tensor, ensure_feasible_policy: bool = True
) -> float: ...


@overload
def exploitability_score(
    env_instance: Environment,
    pi: list[torch.Tensor],
    ensure_feasible_policy: bool = True,
) -> list[float]: ...


def exploitability_score(
    env_instance: Environment,
    pi: torch.Tensor | list[torch.Tensor],
    ensure_feasible_policy: bool = True,
    precision: float | None = None,
) -> float | list[float]:
    """Compute the exploitability metric for a given policy (or policies)."""
    if isinstance(pi, list):
        return [exploitability_score(env_instance, x) for x in pi]

    if ensure_feasible_policy:
        pi = postprocess_policy(env_instance, pi)

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
    expl = cast(float, expl.item())

    # ensure getting non-negative value getting a negative expl but within precision
    expl = 0 if precision is not None and abs(expl) <= precision and expl < 0 else expl

    return expl
