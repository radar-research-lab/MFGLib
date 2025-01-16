from __future__ import annotations

from typing import Callable, cast

import torch

from mfglib import __TORCH_FLOAT__

from mfglib.env import Environment


class QFn:
    """Class for computing Q-values of a given environment."""

    env: Environment
    L: torch.Tensor

    def __init__(
        self, env: Environment, L: torch.Tensor, *, verify_integrity: bool = True
    ) -> None:
        """Initialize a QFn object.

        Args:
        ----
        env: An environment instance
        L: Mean field tensor with shape (T + 1, *S, *A).
        verify_integrity: Optionally verify that L contains valid joint probabilities.
        """
        if verify_integrity:
            for t, L_t in enumerate(L):
                if (L_t < 0).any():
                    raise ValueError(f"negative probability found in time index {t}")
                if (L_t.sum() - 1.0).abs() > 1e-02:
                    raise ValueError(
                        f"joint probability did not sum to 1 in time index {t}"
                    )
        self.env = env
        self.L = L

    def _transition_probabilities(self, t: int) -> torch.Tensor:
        """Compute state-action-state transition probabilities.

        Args
        ----
        t: The current timestep.

        Returns
        -------
        A tensor with dimensions (*S, *A, len(S)).
        """
        l_s = len(self.env.S)
        l_a = len(self.env.A)
        ssa_to_sas = tuple(range(l_s, l_s + l_s + l_a)) + tuple(range(l_s))
        return (
            self.env.prob(t, self.L[t]).permute(ssa_to_sas).flatten(start_dim=l_s + l_a)
        )

    def _compute_q_values(
        self,
        future_rewards_fn: Callable[[int, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Solve for the Q-values.

        Args
        ----
        future_rewards_fn: A callable to compute the future expected rewards. It
            should accept `t: int, q_values: torch.Tensor` where `q_values` has
            shape (T + 1, *S, *A) and return a tensor of shape (*S, *A).

        Returns
        -------
        A tensor with dimensions (T + 1, *S, *A).
        """
        T = self.env.T

        q_values = torch.empty(size=(T + 1, *self.env.S, *self.env.A))
        q_values[T] = self.env.reward(T, self.L[T])

        for t in range(T - 1, -1, -1):
            reward = self.env.reward(t, self.L[t])
            q_values[t] = reward + future_rewards_fn(t, q_values)

        return q_values

    def optimal(self) -> torch.Tensor:
        """Compute optimal Q-values.

        Returns
        -------
        A tensor with dimensions (T + 1, *S, *A).
        """

        def future_rewards(t: int, q_values: torch.Tensor) -> torch.Tensor:
            l_s = len(self.env.S)
            max_q = q_values[t + 1].flatten(start_dim=l_s).max(dim=-1)[0].flatten()
            return self._transition_probabilities(t) @ max_q

        return self._compute_q_values(future_rewards)

    def for_policy(self, pi: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for a given policy.

        Args
        ----
        pi: A tensor with dimensions (T + 1, *S, *A).

        Returns
        -------
        A tensor with dimensions (T + 1, *S, *A).
        """

        def future_rewards(t: int, q_values: torch.Tensor) -> torch.Tensor:
            l_s = len(self.env.S)
            pi_q = (
                (pi[t + 1] * q_values[t + 1])
                .flatten(start_dim=l_s)
                .sum(dim=-1)
                .flatten()
            )
            return self._transition_probabilities(t) @ (pi_q.double() if __TORCH_FLOAT__ == 64 else pi_q.float())

        return self._compute_q_values(future_rewards)
