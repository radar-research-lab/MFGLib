import torch

from mfglib.alg.utils import tuple_prod
from mfglib.env import Environment


def mf_omo_policy(env_instance: Environment, L: torch.Tensor) -> torch.Tensor:
    """Compute the policy given mean-field for MF-OMO.

    Args:
    ----
    env_instance: An instance of a specific environment.
    L: A numpy array/tensor of size (T+1,)+S+A representing the mean-field L.
    """
    # Environment parameters
    S = env_instance.S  # state space dimensions
    A = env_instance.A  # action space dimensions

    # Auxiliary variables
    l_s = len(S)
    l_a = len(A)
    n_a = tuple_prod(A)
    ones_ts = (1,) * (l_s + 1)
    ats_to_tsa = tuple(range(l_a, l_a + 1 + l_s)) + tuple(range(l_a))

    # Corresponding policy

    L_sum_rptd = (
        L.flatten(start_dim=1 + l_s).sum(-1).repeat(A + ones_ts).permute(ats_to_tsa)
    )
    pi = L.div(L_sum_rptd).nan_to_num(
        nan=1 / n_a, posinf=1 / n_a, neginf=1 / n_a
    )  # using uniform distribution when L_t_sum_rptd is zero

    return pi
