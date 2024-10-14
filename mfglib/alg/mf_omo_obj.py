from __future__ import annotations

from typing import Literal

import torch

from mfglib import __TORCH_FLOAT__

from mfglib.alg.mf_omo_params import mf_omo_params
from mfglib.alg.utils import tuple_prod
from mfglib.env import Environment


def mf_omo_obj(
    env_instance: Environment,
    L_u: torch.Tensor,
    z_v: torch.Tensor,
    y_w: torch.Tensor,
    loss: Literal["l1"] | Literal["l2"] | Literal["l1_l2"],
    c1: float,
    c2: float,
    c3: float,
    parameterize: bool,
) -> torch.Tensor:
    """Compute the objective for MF-OMO.

    Args
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
    parameterize: Determines if we use the parameterized setting.

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
    r_max = env_instance.r_max

    # Auxiliary variables
    n_s = tuple_prod(S)
    n_a = tuple_prod(A)
    c_v = n_s * n_a * (T**2 + T + 2) * r_max
    c_w = n_s * (T + 1) * (T + 2) * r_max / 2 / torch.sqrt(torch.tensor(n_s * (T + 1)))

    # Auxiliary functions
    soft_max = torch.nn.Softmax(dim=-1)  # softmax function

    # Get the parameters
    L = (
        soft_max(L_u.flatten(start_dim=1)).reshape((T + 1,) + S + A)
        if parameterize
        else L_u
    )
    z = c_v * soft_max(z_v)[:-1] if parameterize else z_v
    y = c_w * torch.sin(y_w) if parameterize else y_w
    b, A_L, c_L = mf_omo_params(env_instance, L)

    # print(f"{y.dtype=}, {z.dtype=}, {L.dtype=}")

    # Compute the objective
    obj = torch.zeros(1, dtype=torch.float64 if __TORCH_FLOAT__ == 64 else torch.float)
    if loss == "l1":
        obj += c1 * (A_L.matmul(L.flatten().double() if __TORCH_FLOAT__ == 64 else L.flatten().float()) - b).abs().sum()
        obj += c2 * (A_L.transpose(0, 1).matmul(y.double() if __TORCH_FLOAT__ == 64 else y.float()) + z - c_L).abs().sum()
        obj += c3 * z.mul(L.flatten()).sum()
    if loss == "l2":
        obj += c1 * (A_L.matmul(L.flatten().double() if __TORCH_FLOAT__ == 64 else L.flatten().float()) - b).pow(2).sum()
        obj += c2 * (A_L.transpose(0, 1).matmul(y.double() if __TORCH_FLOAT__ == 64 else y.float()) + z - c_L).pow(2).sum()
        obj += c3 * z.mul(L.flatten()).sum().pow(2)
    if loss == "l1_l2":
        obj += c1 * (A_L.matmul(L.flatten().double() if __TORCH_FLOAT__ == 64 else L.flatten().float()) - b).pow(2).sum()
        obj += c2 * (A_L.transpose(0, 1).matmul(y.double() if __TORCH_FLOAT__ == 64 else y.float()) + z - c_L).pow(2).sum()
        obj += c3 * z.mul(L.flatten()).sum()

    return obj
