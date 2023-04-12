from __future__ import annotations

import torch

from mfglib.alg.utils import tuple_prod
from mfglib.env import Environment


def mf_omo_params(
    env_instance: Environment, L: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the required parameters for MF-OMO.

    Args
    ----
    env_instance: An instance of a specific environment.
    L: A tensor of size (T+1,)+S+A with tracked gradient representing
        the mean-field L.

    Notes
    -----
    See [1]_ for details.

    .. [1] MF-OMO: An Optimization Formulation of Mean-Field Games
    Guo, X., Hu, A., & Zhang, J. (2022). arXiv:2206.09608.
    """
    # Environment parameters
    T = env_instance.T  # time horizon
    S = env_instance.S  # state space dimensions
    A = env_instance.A  # action space dimensions
    mu0 = env_instance.mu0  # initial state distribution

    # Auxiliary variables
    l_s = len(S)
    n_s = tuple_prod(S)
    n_a = tuple_prod(A)
    v = torch.zeros((n_s, n_a))
    v[0, :] = 1.0
    z = v.clone()  # matrix z with a different column arrangement
    for i in range(1, n_s):
        z = torch.cat((z, torch.roll(v, i, 0)), dim=1)

    # Vector b
    b = torch.zeros((T + 1,) + S)
    b[T] = mu0
    b = b.flatten()

    # Matrix A_L and vector c_L
    A_L = torch.zeros((T + 1) * n_s, (T + 1) * n_s * n_a)
    c_L = torch.zeros(((T + 1) * n_s * n_a,))
    for t in range(T):
        p_t = env_instance.prob(t, L[t]).flatten(start_dim=l_s).flatten(end_dim=-2)
        r_t = env_instance.reward(t, L[t]).flatten()
        A_L[t * n_s : (t + 1) * n_s, t * n_s * n_a : (t + 1) * n_s * n_a] += p_t
        A_L[t * n_s : (t + 1) * n_s, (t + 1) * n_s * n_a : (t + 2) * n_s * n_a] -= z
        c_L[t * n_s * n_a : (t + 1) * n_s * n_a] -= r_t
    A_L[T * n_s : (T + 1) * n_s, 0 : n_s * n_a] += z
    c_L[T * n_s * n_a : (T + 1) * n_s * n_a] -= env_instance.reward(T, L[T]).flatten()

    return b, A_L, c_L
