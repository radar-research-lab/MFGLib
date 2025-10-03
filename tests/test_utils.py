from __future__ import annotations

import numpy as np
import pytest
import torch

from mfglib.alg import OnlineMirrorDescent
from mfglib.alg.utils import (
    hat_initialization,
    project_onto_simplex,
)
from mfglib.env import Environment
from mfglib.utils import mean_field_from_policy, policy_from_mean_field


def test_project_onto_simplex() -> None:
    x = torch.tensor([0.0, 2.0, 0.0])
    projection = project_onto_simplex(x)
    assert isinstance(projection, torch.Tensor)
    assert torch.allclose(projection, torch.tensor([0.0, 1.0, 0.0]), rtol=0, atol=1e-5)

    y = torch.tensor(2.0)
    projection = project_onto_simplex(y)
    assert isinstance(projection, torch.Tensor)
    assert torch.allclose(projection, torch.tensor([1.0]), rtol=0, atol=1e-5)

    z = torch.tensor([0.0, -2.0])
    projection = project_onto_simplex(z)
    assert isinstance(projection, torch.Tensor)
    assert torch.allclose(projection, torch.tensor([1.0, 0.0]), rtol=0, atol=1e-5)


@pytest.mark.parametrize(
    "env",
    [
        Environment.left_right(),
        Environment.susceptible_infected(),
        Environment.rock_paper_scissors(),
    ],
)
def test_hat_initialization(env: Environment) -> None:
    size = (env.T + 1, *env.S, *env.A)

    def tuple_prod(tup: tuple[int, ...]) -> int:
        return np.prod(tup).item()

    L = torch.ones(size=size) / np.prod(env.S + env.A).item()

    z_hat, y_hat = hat_initialization(env, L, parameterize=False)
    assert isinstance(z_hat, torch.Tensor)
    assert isinstance(y_hat, torch.Tensor)
    assert z_hat.shape == (tuple_prod(size),)
    assert y_hat.shape == (tuple_prod((env.T + 1, *env.S)),)
    assert (z_hat >= 0).all()

    v_hat, w_hat = hat_initialization(env, L, parameterize=True)
    if v_hat is not None:
        assert isinstance(v_hat, torch.Tensor)
        assert v_hat.shape == (tuple_prod(size) + 1,)
    assert isinstance(w_hat, torch.Tensor)
    assert w_hat.shape == (tuple_prod((env.T + 1, *env.S)),)


@pytest.mark.parametrize(
    "env",
    [
        Environment.left_right(),
        Environment.rock_paper_scissors(),
        Environment.susceptible_infected(),
        Environment.conservative_treasure_hunting(),
        Environment.equilibrium_price(),
    ],
)
def test_policy_mean_field_consistency(env: Environment) -> None:
    """Verify that the policy is consistent with the mean-field."""
    algorithm = OnlineMirrorDescent()  # chosen arbitrarily

    sols, _, _ = algorithm.solve(env)
    pi = sols[-1]
    L = mean_field_from_policy(pi, env=env)

    l_s = len(env.S)
    l_a = len(env.A)
    dim = tuple(range(l_s, l_s + l_a))

    for t in range(env.T + 1):
        conditional = L[t] / L[t].sum(dim=dim, keepdim=True)
        mask = ~conditional.isnan()
        assert torch.allclose(
            conditional.masked_select(mask),
            pi[t].masked_select(mask),
        )


def test_mean_field_on_lr() -> None:
    left_right = Environment.left_right()

    pi = torch.tensor(
        [
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        ],
    )
    L = mean_field_from_policy(pi, env=left_right)

    assert torch.allclose(
        L[0],
        torch.tensor(
            [
                [0.5, 0.5],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
        ),
    )

    assert torch.allclose(
        L[1],
        torch.tensor(
            [
                [0.0, 0.0],
                [0.25, 0.25],
                [0.25, 0.25],
            ],
        ),
    )
