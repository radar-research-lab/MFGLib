from __future__ import annotations

import numpy as np
import pytest
import torch

from mfglib.alg.utils import hat_initialization, project_onto_simplex, extract_policy_from_mean_field
from mfglib.env import Environment
from mfglib.mean_field import mean_field


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


def test_mean_field_policy_roundtrip() -> None:
    sis = Environment.susceptible_infected(T=2, mu0=(0.6, 0.4))

    # fmt: off
    L = torch.tensor(
        [[[0.0000, 0.6259], [0.3741, 0.0000]],
         [[0.7766, 0.0000], [0.2234, 0.0000]],
         [[0.7144, 0.0000], [0.2856, 0.0000]]]
    )
    # fmt: on
    pi = extract_policy_from_mean_field(sis, L)

    L_new = mean_field(sis, pi)
    assert torch.allclose(L_new, L)
