from __future__ import annotations

from typing import Any, Literal

import pytest
import torch
from torch.testing import assert_close

from mfglib.alg import MFOMO, FictitiousPlay, OnlineMirrorDescent, PriorDescent
from mfglib.alg.abc import Algorithm
from mfglib.env import Environment


def _assert_lr_solved(
    algorithm: Algorithm, lr: Environment, atol: float, rtol: float
) -> None:
    """See https://github.com/junziz/MFGLib/issues/55 for derivations."""
    solns, _, _ = algorithm.solve(lr, max_iter=2000, atol=atol, rtol=rtol)
    pi = solns[-1]

    # The shape should reflect two time steps, three states, and two actions
    assert pi.shape == (2, 3, 2)

    # NE requirement as derived in GH#55
    result = lr.mu0 @ pi[0, :, 0]
    expected = (2.0 / 3.0) * torch.ones_like(result)
    assert_close(result, expected, atol=atol, rtol=rtol)

    # Policies at all time steps should be valid probability distributions
    result = pi.sum(dim=-1)
    expected = torch.ones_like(result)
    assert_close(result, expected)


@pytest.mark.parametrize(
    "algorithm",
    [
        FictitiousPlay(alpha=0.001),
        PriorDescent(eta=1.0, n_inner=50),
        OnlineMirrorDescent(alpha=1.0),
    ],
)
@pytest.mark.parametrize(
    "mu0",
    [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.25, 0.25),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
    ],
)
@pytest.mark.parametrize("atol", [1e-4])
@pytest.mark.parametrize("rtol", [1e-4])
def test_algorithms_on_lr(
    algorithm: Algorithm, mu0: tuple[float, float, float], atol: float, rtol: float
) -> None:
    _assert_lr_solved(
        algorithm=algorithm, lr=Environment.left_right(mu0), atol=atol, rtol=rtol
    )


@pytest.mark.parametrize(
    "source,name,config,n_iter,parameterized",
    [
        ("pytorch", "Adam", {"lr": 1.0}, 300, False),
        ("pytorch", "Adam", {"lr": 0.1}, 1000, True),
    ],
)
def test_mf_omo_on_lr(
    source: Literal["pytorch"],
    name: str,
    config: dict[str, Any],
    n_iter: int,
    parameterized: bool,
) -> None:
    lr = Environment.left_right(mu0=(1.0, 0.0, 0.0))
    algorithm = MFOMO(
        optimizer={"source": source, "name": name, "config": config},
    )
    _assert_lr_solved(algorithm=algorithm, lr=lr, atol=0.0005, rtol=0.00008)


@pytest.mark.parametrize(
    "alg",
    [
        FictitiousPlay(),
        MFOMO(),
        OnlineMirrorDescent(),
        PriorDescent(),
    ],
)
@pytest.mark.parametrize("stat", ["iter", "rt", "expl"])
def test_tuner_on_rps(alg: Algorithm, stat: Literal["iter", "rt", "expl"]) -> None:
    rps = Environment.rock_paper_scissors()

    alg.tune_on_failure_rate(
        envs=[rps],
        stat=stat,
        fail_thresh=100,
        max_iter=500,
        atol=0,
        rtol=1e-1,
        n_trials=5,
        timeout=20,
    )
    alg.tune_on_geometric_mean(
        envs=[rps],
        stat=stat,
        shift=10,
        max_iter=500,
        atol=0,
        rtol=1e-1,
        n_trials=5,
        timeout=20,
    )
