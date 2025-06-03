from __future__ import annotations

from tempfile import TemporaryDirectory
from typing import Any, Literal

import pytest
import torch
from torch.testing import assert_close

from mfglib.alg import (
    MFOMO,
    FictitiousPlay,
    OccupationMeasureInclusion,
    OnlineMirrorDescent,
    PriorDescent,
)
from mfglib.alg.abc import Algorithm
from mfglib.env import Environment
from mfglib.tuning import FailureRate, GeometricMean


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

    alg.tune(
        envs=[rps],
        pi_0s="uniform",
        metric=FailureRate(fail_thresh=100, stat=stat),
        solve_kwargs={"max_iter": 200},
        n_trials=5,
        timeout=20,
    )
    # NOTE: fail_thresh is not required when stat == "rt"
    if stat != "rt":
        alg.tune(
            envs=[rps],
            pi_0s="uniform",
            metric=FailureRate(stat=stat),
            solve_kwargs={"max_iter": 200},
            n_trials=5,
            timeout=20,
        )
    alg.tune(
        envs=[rps],
        pi_0s="uniform",
        metric=GeometricMean(shift=1.0, stat=stat),
        solve_kwargs={"max_iter": 200},
        n_trials=5,
        timeout=20,
    )


@pytest.mark.parametrize("alpha", [None, 1.0])
def test_fictitious_play_save_and_load(alpha: float | None) -> None:
    fp = FictitiousPlay(alpha=alpha)
    with TemporaryDirectory() as tmpdir:
        fp.save(tmpdir)
        fp2 = FictitiousPlay.load(tmpdir)
        assert fp2.alpha == fp.alpha


@pytest.mark.parametrize("eta", [0.5])
@pytest.mark.parametrize("n_inner", [None, 50])
def test_prior_descent_save_and_load(eta: float, n_inner: int | None) -> None:
    pd = PriorDescent(eta=eta, n_inner=n_inner)
    with TemporaryDirectory() as tmpdir:
        pd.save(tmpdir)
        pd2 = PriorDescent.load(tmpdir)
        assert pd2.eta == pd.eta
        assert pd2.n_inner == pd.n_inner


@pytest.mark.parametrize("alpha", [2.0])
def test_mirror_descent_save_and_load(alpha: float) -> None:
    md = OnlineMirrorDescent()
    with TemporaryDirectory() as tmpdir:
        md.save(tmpdir)
        md2 = OnlineMirrorDescent.load(tmpdir)
        assert md2.alpha == md.alpha


def test_mf_omo_save_and_load() -> None:
    mf = MFOMO(c1=4.0, loss="l1")
    with TemporaryDirectory() as tmpdir:
        mf.save(tmpdir)
        mf2 = MFOMO.load(tmpdir)
        assert mf2.c1 == mf.c1
        assert mf2.loss == mf.loss


def test_mf_omi_save_and_load() -> None:
    mf = OccupationMeasureInclusion(alpha=0.1, eta=2.0)
    with TemporaryDirectory() as tmpdir:
        mf.save(tmpdir)
        mf2 = OccupationMeasureInclusion.load(tmpdir)
        assert mf2.alpha == mf.alpha
        assert mf2.eta == mf.eta
