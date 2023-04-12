import pytest
import torch
from torch.testing import assert_close

from mfglib.alg import OnlineMirrorDescent
from mfglib.env import Environment
from mfglib.mean_field import mean_field


@pytest.mark.parametrize(
    "environment",
    [
        Environment.left_right(),
        Environment.rock_paper_scissors(),
        Environment.susceptible_infected(),
        Environment.conservative_treasure_hunting(),
        Environment.equilibrium_price(),
    ],
)
def test_policy_mean_field_consistency(environment: Environment) -> None:
    """Verify that the policy is consistent with the mean-field."""
    algorithm = OnlineMirrorDescent()  # chosen arbitrarily

    sols, _, _ = algorithm.solve(environment)
    pi = sols[-1]
    L = mean_field(environment, pi)

    l_s = len(environment.S)
    l_a = len(environment.A)
    dim = tuple(range(l_s, l_s + l_a))

    for t in range(environment.T + 1):
        conditional = L[t] / L[t].sum(dim=dim, keepdim=True)
        mask = ~conditional.isnan()
        assert_close(
            conditional.masked_select(mask),
            pi[t].masked_select(mask),
            check_dtype=False,
        )


def test_mean_field_on_lr() -> None:
    left_right = Environment.left_right()

    pi = torch.tensor(
        [
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        ],
    )
    result = mean_field(left_right, pi)

    assert_close(
        actual=result[0],
        expected=torch.tensor(
            [
                [0.5, 0.5],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
        ),
    )

    assert_close(
        actual=result[1],
        expected=torch.tensor(
            [
                [0.0, 0.0],
                [0.25, 0.25],
                [0.25, 0.25],
            ],
        ),
    )
