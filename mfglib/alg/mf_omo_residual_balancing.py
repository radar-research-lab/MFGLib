from __future__ import annotations

from typing import Literal

import torch

from mfglib.alg.mf_omo_params import mf_omo_params
from mfglib.env import Environment


def mf_omo_residual_balancing(
    env_instance: Environment,
    L_u: torch.Tensor,
    z_v: torch.Tensor,
    y_w: torch.Tensor,
    loss: Literal["l1"] | Literal["l2"] | Literal["l1_l2"],
    c1: float,
    c2: float,
    m1: float,
    m2: float,
    m3: float,
    parameterize: bool,
    safeguard: float = 1e3,
) -> tuple[float, float]:
    """Apply adaptive residual balancing.

    Args:
    ----
    env_instance: An instance of a specific environment.
    L_u: A tensor of size (T+1,)+S+A representing the mean-field L or its
    parameterized version u.
    z_v: A one dimensional tensor of size (T+1)*|S|*|A| ((T+1)*|S|*|A|+1)
    representing the slack variable z (parameterized version of z, v).
    y_w: A one dimensional tensor of size (T+1)*|S| representing the
        variable y or its parameterized version w.
    loss: Determines the type of norm used in the objective function.
    c1: Redesigned objective coefficient.
    c2: Redesigned objective coefficient.
    c3: Redesigned objective coefficient.
    m1: Residual balancing parameter.
    m2: Residual balancing parameter.
    m3: Residual balancing parameter.
    parameterize: Determines if we use the parameterized setting.
    safeguard: Upperbound for redesigned objective coefficients.
    """
    # Environment parameters
    T = env_instance.T  # time horizon
    S = env_instance.S  # state space dimensions
    A = env_instance.A  # action space dimensions
    r_max = env_instance.r_max

    # Auxiliary variables
    n_s = env_instance.n_states
    n_a = env_instance.n_states
    c_v = n_s * n_a * (T**2 + T + 2) * r_max
    c_w = n_s * (T + 1) * (T + 2) * r_max / 2 / torch.sqrt(torch.tensor(n_s * (T + 1)))

    # Auxiliary functions
    soft_max = torch.nn.Softmax(dim=-1)  # softmax function

    # Three terms in the objective function
    L = L_u.clone().detach()
    z = z_v.clone().detach()
    y = y_w.clone().detach()
    if parameterize:
        L = soft_max(L.flatten(start_dim=1)).reshape((T + 1,) + S + A)
        z = c_v * soft_max(z)[:-1]
        y = c_w * torch.sin(y)
    b, A_L, c_L = mf_omo_params(env_instance, L)

    if loss == "l1":
        o1 = (A_L.matmul(L.flatten()) - b).abs().sum()
        o2 = (A_L.transpose(0, 1).matmul(y) + z - c_L).abs().sum()
        o3 = z.mul(L.flatten()).sum()
    if loss == "l2":
        o1 = (A_L.matmul(L.flatten()) - b).pow(2).sum()
        o2 = (A_L.transpose(0, 1).matmul(y) + z - c_L).pow(2).sum()
        o3 = z.mul(L.flatten()).sum().pow(2)
    if loss == "l1_l2":
        o1 = (A_L.matmul(L.flatten()) - b).pow(2).sum()
        o2 = (A_L.transpose(0, 1).matmul(y) + z - c_L).pow(2).sum()
        o3 = z.mul(L.flatten()).sum()

    # Apply residual balancing
    c1_new = c1
    c2_new = c2
    if torch.maximum(o2, o3) != 0.0:
        if o1 / torch.maximum(o2, o3) > m1:
            c1_new = c1 * m2
    if torch.minimum(o2, o3) != 0.0:
        if o1 / torch.minimum(o2, o3) < m3:
            c1_new = c1 / m2
    if torch.maximum(o1, o3) != 0.0:
        if o2 / torch.maximum(o1, o3) > m1:
            c2_new = c2 * m2
    if torch.minimum(o1, o3) != 0.0:
        if o2 / torch.minimum(o1, o3) < m3:
            c2_new = c2 / m2

    return min(c1_new, safeguard), min(c2_new, safeguard)
