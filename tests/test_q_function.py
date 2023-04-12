import pytest
import torch
from torch.testing import assert_close

from mfglib.alg.q_fn import QFn
from mfglib.alg.utils import tuple_prod
from mfglib.env import Environment


@pytest.mark.parametrize(
    "env",
    [
        Environment.left_right(),
        Environment.conservative_treasure_hunting(),
        Environment.susceptible_infected(),
        Environment.rock_paper_scissors(),
        Environment.equilibrium_price(),
    ],
)
def test_q_function(env: Environment) -> None:
    """Primarily a property-based test."""
    size = (env.T + 1, *env.S, *env.A)

    L = torch.ones(size=size) / tuple_prod(env.S + env.A)
    pi = torch.ones(size=size) / tuple_prod(env.A)

    optimal = QFn(env, L).optimal()
    assert optimal.size() == torch.Size(size)
    assert optimal.isfinite().all().item()

    for_policy = QFn(env, L).for_policy(pi)
    assert for_policy.size() == torch.Size(size)
    assert for_policy.isfinite().all().item()

    with pytest.raises(ValueError):
        QFn(env, torch.ones(size=size))


def test_q_function_on_lr() -> None:
    lr = Environment.left_right()

    zeros = torch.zeros((2, 3, 2))

    # If the mean-field is in the Center state for all t, then the
    # optimal Q-values should be zero
    L = torch.tensor(
        [
            [[0.5, 0.5], [0.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ],
    )
    assert_close(QFn(lr, L).optimal(), zeros)
    L = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.5, 0.5], [0.0, 0.0], [0.0, 0.0]],
        ],
    )
    assert_close(QFn(lr, L).optimal(), zeros)
    L = torch.tensor(
        [
            [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        ],
    )
    assert_close(QFn(lr, L).optimal(), zeros)

    # The entire mean-field goes Left at t = 0 and Right at t = 1
    L = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        ],
    )
    expected = torch.tensor(
        [
            [[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]],
            [[0.0, 0.0], [-1.0, -1.0], [0.0, 0.0]],
        ]
    )
    assert_close(QFn(lr, L).optimal(), expected)

    # For the above mean-field, the optimal response policy
    # is to choose Right at t = 0.
    pi = torch.tensor(
        [
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        ]
    )
    assert_close(QFn(lr, L).for_policy(pi), expected)
