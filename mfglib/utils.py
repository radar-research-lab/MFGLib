from __future__ import annotations

import torch

from mfglib.env import Environment


def mean_field_from_policy(pi: torch.Tensor, *, env: Environment) -> torch.Tensor:
    """Compute the mean-field L = Γ(π) corresponding to the policy π.

    Args
    ----
    pi: A tensor with dimensions (T+1, *S, *A) where each time slice represents
        a *conditional* probability distribution over actions given state.
    env: An environment instance.

    Returns
    -------
    A tensor with dimensions (T+1 *S, *A) where each time slice represents a
    *joint* probability distribution over states and actions.
    """
    L = torch.empty(size=[env.T + 1, env.n_states, env.n_actions])

    S_dim = len(env.S)

    pi = pi.flatten(start_dim=1, end_dim=S_dim).flatten(start_dim=2)

    mu0 = env.mu0.flatten()
    L[0] = torch.einsum("s,sa->sa", mu0, pi[0])

    for t in range(env.T):
        Lₜ = L[t].reshape(*env.S, *env.A)
        Pₜ = (
            env.prob(t, Lₜ)
            .flatten(end_dim=S_dim - 1)
            .flatten(start_dim=1, end_dim=S_dim)
            .flatten(start_dim=2)
        )
        L[t + 1] = torch.einsum("sa,SA,sSA->sa", pi[t + 1], L[t], Pₜ)

    return L.reshape(env.T + 1, *env.S, *env.A)


def policy_from_mean_field(
    L: torch.Tensor, *, env: Environment, tol: float | None = None
) -> torch.Tensor:
    """Compute the policy π = Π(L) corresponding to the mean field L.

    Args:
        L: A tensor with dimensions (T+1, *S, *A) where each time slice represents
            a *joint* probability distribution over states and actions.
        env: An environment instance.
        tol: Optionally set `L[L.abs() <= tol] = 0` to avoid numerical blowup.

    Returns:
    A tensor with dimensions (T+1, *S, *A) where each time slice represents
        a *conditional* probability distribution over actions given state.
    """
    S_dim = len(env.S)
    L = L.flatten(start_dim=1, end_dim=S_dim).flatten(start_dim=2)

    if tol is not None:
        L[L.abs() <= tol] = 0

    pi = L / L.sum(dim=-1, keepdim=True)
    pi = pi.nan_to_num(nan=1 / env.n_actions)

    return pi.reshape(env.T + 1, *env.S, *env.A)
