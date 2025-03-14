import torch

from mfglib.env import Environment
from mfglib.q_fn import QFn


def Greedy_Policy(env_instance: Environment, L: torch.Tensor) -> torch.Tensor:
    """Compute the greedy policy pi_s corresponding to a mean-field L."""
    # Environment parameters
    T = env_instance.T  # time horizon
    S = env_instance.S  # state space dimensions
    A = env_instance.A  # action space dimensions

    # Auxiliary variables
    l_s = len(S)
    l_a = len(A)
    ones_s = (1,) * l_s
    as_to_sa = tuple(range(l_a, l_a + l_s)) + tuple(range(l_a))

    # Compute Q_s_L
    Q_s_L = QFn(env_instance, L).optimal()

    # Greedy policy
    pi_s = []
    for t in range(T + 1):
        max_per_state_Q_s_L, _ = Q_s_L[t].flatten(start_dim=l_s).max(dim=-1)
        mask = Q_s_L[t] == max_per_state_Q_s_L.repeat(A + ones_s).permute(as_to_sa)
        grdy_pi = torch.ones(S + A).mul(mask)
        grdy_pi_sum_rptd = (
            grdy_pi.flatten(start_dim=l_s).sum(-1).repeat(A + ones_s).permute(as_to_sa)
        )
        pi_s.append(grdy_pi.div(grdy_pi_sum_rptd))

    return torch.stack(pi_s)
