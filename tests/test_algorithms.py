from __future__ import annotations

from tempfile import TemporaryDirectory
from typing import Literal

import pytest
import torch

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


@pytest.mark.parametrize(
    "alg",
    [
        FictitiousPlay(alpha=0.001),
        PriorDescent(eta=2.0, n_inner=50),
        OnlineMirrorDescent(alpha=0.01),
        MFOMO(),
        OccupationMeasureInclusion(),
    ],
)
@pytest.mark.parametrize(
    "mu0",
    [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1 / 3, 1 / 3, 1 / 3),
    ],
)
def test_alg_on_lr(alg: Algorithm, mu0: tuple[float, float, float]) -> None:
    """Algorithm solves simple Left Right environment.

    See https://github.com/junziz/MFGLib/issues/55 for NE derivation.
    """
    lr = Environment.left_right(mu0)

    pis, _, _ = alg.solve(lr, max_iter=1000, atol=None, rtol=None)
    pi_last = pis[-1]

    # The shape should reflect two time steps, three states, and two actions
    assert pi_last.shape == (2, 3, 2)

    # NE requirement as derived in GH#55
    result = lr.mu0 @ pi_last[0, :, 0]
    expected = (2.0 / 3.0) * torch.ones_like(result)
    torch.testing.assert_close(result, expected, atol=5e-4, rtol=5e-4)

    # Policies at all time steps should be valid probability distributions
    result = pi_last.sum(dim=-1)
    expected = torch.ones_like(result)
    torch.testing.assert_close(result, expected)


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
        pi0s="uniform",
        metric=FailureRate(fail_thresh=100, stat=stat),
        solve_kwargs={"max_iter": 200},
        n_trials=5,
        timeout=20,
    )
    # NOTE: fail_thresh is not required when stat == "rt"
    if stat != "rt":
        alg.tune(
            envs=[rps],
            pi0s="uniform",
            metric=FailureRate(stat=stat),
            solve_kwargs={"max_iter": 200},
            n_trials=5,
            timeout=20,
        )
    alg.tune(
        envs=[rps],
        pi0s="uniform",
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
