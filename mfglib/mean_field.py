import torch

from mfglib.env import Environment


def mean_field(env: Environment, pi: torch.Tensor) -> torch.Tensor:
    """Compute the mean-field gamma_pi corresponding to the policy pi.

    Args
    ----
    env: An Environment instance.
    pi: A tensor with dimensions (T + 1, *S, *A) where each time slice represents
        a *conditional* probability distribution.

    Returns
    -------
    A tensor with dimensions (T + 1 *S, *A) where each time slice represents a
    *joint* probability distribution.
    """
    T = env.T
    A = env.A
    mu0 = env.mu0

    l_s = len(env.S)
    l_a = len(env.A)

    ones_s = (1,) * l_s
    as_to_sa = tuple(range(l_a, l_a + l_s)) + tuple(range(l_a))

    gamma_pi = torch.empty(size=(T + 1, *env.S, *env.A))
    gamma_pi[0] = mu0.repeat(A + ones_s).permute(as_to_sa).mul(pi[0])

    for t in range(1, T + 1):
        prob_prev_fltn = env.prob(t - 1, gamma_pi[t - 1]).flatten(start_dim=l_s)
        gamma_pi_prev_fltn = gamma_pi[t - 1].flatten()
        s = prob_prev_fltn.matmul(gamma_pi_prev_fltn.float())
        s_rptd = s.repeat(A + ones_s).permute(as_to_sa)
        gamma_pi[t] = pi[t].mul(s_rptd)

    return gamma_pi
