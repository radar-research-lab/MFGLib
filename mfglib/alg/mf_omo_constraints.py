from __future__ import annotations

import torch

from mfglib.alg.utils import project_onto_simplex, tuple_prod
from mfglib.env import Environment


def mf_omo_constraints(
    env_instance: Environment,
    L: torch.Tensor,
    z: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enforce the required constraints for MF-OMO.

    Args
    ----
    env_instance: An instance of a specific environment.
    L: A tensor of size (T+1,)+S+A representing the mean-field L.
    z: A one dimensional tensor of size (T+1)*|S|*|A| representing
        the slack variable z.
    y: A one dimensional tensor of size (T+1)*|S| representing the
        variable y.

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
    r_max = env_instance.r_max  # reward supremum

    # Auxiliary variables
    n_s = tuple_prod(S)
    n_a = tuple_prod(A)

    # Clone and detach tensors
    L_p = L.clone().detach()
    z_p = z.clone().detach()
    y_p = y.clone().detach()

    # Project L
    for t in range(T + 1):
        L_p[t] = project_onto_simplex(L_p[t].flatten()).reshape(S + A)

    # Project z
    c_z = n_s * n_a * (T**2 + T + 2) * r_max
    p_ind = torch.argwhere(z_p > 0).ravel()
    np_ind = torch.argwhere(z_p <= 0).ravel()
    if len(p_ind) > 0 and z_p[p_ind].sum() > c_z:
        z_p[p_ind] = project_onto_simplex(z_p[p_ind], r=c_z)
    z_p[np_ind] = 0.0

    # Normalize y
    c_y = n_s * (T + 1) * (T + 2) * r_max / 2
    y_norm = torch.sqrt(y.pow(2).sum())
    if y_norm > c_y:
        y_p = (y_p / y_norm) * c_y

    return L_p, z_p, y_p
