from __future__ import annotations

from functools import partial
from typing import Any, Callable

import torch

from mfglib.env import Environment


def _stationary(environment: Environment, data: Any, **kwargs: Any) -> torch.Tensor:
    """Stationary policy constructor, for internal use only."""
    tensor = torch.tensor(data, **kwargs)
    return torch.stack([tensor for _ in range(environment.T + 1)])


def _uniform(environment: Environment, **kwargs: Any) -> torch.Tensor:
    """Uniform policy constructor, for internal use only."""
    t, s, a = environment.T, environment.S, environment.A
    return torch.ones(t + 1, *s, *a, **kwargs) / torch.ones(a, **kwargs).sum()


def _greedy(environment: Environment, L: torch.Tensor) -> torch.Tensor:
    """Greedy policy constructor, for internal use only."""
    from mfglib.q_fn import QFn

    T, S, A = environment.T, environment.S, environment.A

    # Auxiliary variables
    l_S = len(S)
    l_A = len(A)
    ones_S = (1,) * l_S
    AS_to_SA = tuple(range(l_A, l_A + l_S)) + tuple(range(l_A))

    # Compute Q_s_L
    Q_s_L = QFn(environment, L).optimal()

    # Greedy policy
    pi_s = torch.empty(T + 1, *S, *A)
    for t in range(T + 1):
        max_per_state_Q_s_L, _ = Q_s_L[t].flatten(start_dim=l_S).max(dim=-1)
        mask = Q_s_L[t] == max_per_state_Q_s_L.repeat(A + ones_S).permute(AS_to_SA)
        grdy_pi = torch.ones(S + A).mul(mask)
        grdy_pi_sum_rptd = (
            grdy_pi.flatten(start_dim=l_S).sum(-1).repeat(A + ones_S).permute(AS_to_SA)
        )
        pi_s[t] = grdy_pi.div(grdy_pi_sum_rptd)

    return pi_s


class Policy:
    """Encode a conditional probability distribution at all (T, *S, *A) points.

    In general, a ``Policy`` can be highly multidimensional (greater than three) since
    both states and actions are permitted to be multidimensional themselves. For
    single-dimensional states and actions, the index (i, j, k) of the policy can be
    thought of as the probability of taking action k conditioned on being in state j
    at time i.

    The ``Policy`` class represents a delayed policy computation. That is to say, the
    actual numeric policy is encoded by a ``torch.Tensor`` and must be retrieved via
    ``Policy.build``. This design allows us to delay the coupling of a ``Policy`` with
    an ``Environment`` as late as possible.
    """

    generator: Callable[[Environment], torch.Tensor]

    def __init__(self, data: Any, **kwargs: Any) -> None:
        """Pass data and kwargs to torch.tensor constructor."""
        fn = partial(torch.tensor, data=data, **kwargs)
        self.generator = lambda _: fn()  # noqa: E731

    @classmethod
    def stationary(cls, data: Any, **kwargs: Any) -> Policy:
        """Broadcast the same policy across all time-steps."""
        policy = cls.__new__(cls)
        policy.generator = partial(_stationary, data=data, **kwargs)
        return policy

    @classmethod
    def uniform(cls, **kwargs: Any) -> Policy:
        """Assign uniform probabilities to all actions at a given (T, *S) point."""
        policy = cls.__new__(cls)
        policy.generator = partial(_uniform, **kwargs)
        return policy

    @classmethod
    def greedy(cls, L: torch.Tensor) -> Policy:
        """Compute the greedy policy corresponding to a mean-field L."""
        policy = cls.__new__(cls)
        policy.generator = partial(_greedy, L=L)
        return policy

    def build(
        self, environment: Environment, *, verify_integrity: bool = True
    ) -> torch.Tensor:
        """Build a conditional distribution with shape (T + 1, *S, *A)."""
        policy = self.generator(environment)
        if verify_integrity:
            if (policy < 0).any():
                raise ValueError("negative probability found in policy")
            if policy.shape != (environment.T + 1, *environment.S, *environment.A):
                raise ValueError(
                    f"policy dimensions {policy.shape} don't match environment"
                )
            action_dims = tuple(-i - 1 for i, _ in enumerate(environment.A))
            if ((policy.sum(dim=action_dims) - 1.0).abs() > 1e-8).any():
                raise ValueError(
                    "conditional distribution did not sum to 1 across actions"
                )
        return policy
