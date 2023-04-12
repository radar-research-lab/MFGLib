from __future__ import annotations

import math

import pytest
import torch

from mfglib.alg.utils import (
    failure_rate,
    hat_initialization,
    project_onto_simplex,
    shifted_geometric_mean,
    tuple_prod,
)
from mfglib.env import Environment


@pytest.mark.parametrize("num_type", [int, float])
def test_tuple_prod(num_type: type[int] | type[float]) -> None:
    """Check that the result is correct, and the datatype is preserved."""
    tup = tuple(num_type(x) for x in range(1, 4))
    result = tuple_prod(tup)
    assert result == num_type(6)
    assert isinstance(result, num_type)


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

    L = torch.ones(size=size) / tuple_prod(env.S + env.A)

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


def test_shifted_geometric_mean() -> None:
    x = [100]
    sgm = shifted_geometric_mean(x)
    assert sgm == 100.0

    y = [0, 990]
    sgm = shifted_geometric_mean(y)
    assert math.isclose(sgm, 90.0, rel_tol=0, abs_tol=1e-5)


def test_failure_rate() -> None:
    x = [100]
    fr = failure_rate(x, 10.0)
    assert fr == 1.0

    y = [0, 10, 100, 1000]
    fr = failure_rate(y, 50.0)
    assert math.isclose(fr, 0.5, rel_tol=0, abs_tol=1e-5)
