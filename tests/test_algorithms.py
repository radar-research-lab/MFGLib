from __future__ import annotations

from tempfile import TemporaryDirectory

import numpy as np
import optuna
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
from mfglib.tuning import GeometricMean

optuna.logging.set_verbosity(optuna.logging.ERROR)


@pytest.mark.parametrize(
    "alg",
    [
        FictitiousPlay(alpha=0.01),
        PriorDescent(eta=2.0, n_inner=60),
        OnlineMirrorDescent(alpha=0.01),
        MFOMO(),
        OccupationMeasureInclusion(alpha=0.01),
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

    pis, _, expls = alg.solve(lr, max_iter=4000, atol=None, rtol=None)
    assert len(pis) == len(expls)
    pi_last = pis[-1]

    # The shape should reflect two time steps, three states, and two actions
    assert pi_last.shape == (2, 3, 2)

    # NE requirement as derived in GH#55
    result = lr.mu0 @ pi_last[0, :, 0]
    np.testing.assert_approx_equal(result, 2.0 / 3.0, significant=2)

    # Policies at all time steps should be valid probability distributions
    result = pi_last.sum(dim=-1)
    expected = torch.ones_like(result)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    "alg",
    [
        FictitiousPlay(),
        OnlineMirrorDescent(),
        PriorDescent(),
        OccupationMeasureInclusion(),
    ],
)
def test_tuner_reduces_expl(alg: Algorithm) -> None:
    """Tuner workflow reduces minimal exploitability."""
    MAX_ITER = 20

    env = Environment.susceptible_infected(T=7)

    optuna_study = alg.tune(
        metric=GeometricMean(shift=0),
        envs=[env],
        solve_kwargs={"max_iter": MAX_ITER, "atol": None, "rtol": None},
        n_trials=20,
    )
    alg_tune = alg.from_study(optuna_study)

    _, expls_orig, _ = alg.solve(env, max_iter=MAX_ITER, atol=None, rtol=None)
    _, expls_tune, _ = alg_tune.solve(env, max_iter=MAX_ITER, atol=None, rtol=None)

    min_expl_orig = min(expls_orig)
    min_expl_tune = min(expls_tune)

    assert optuna_study.best_trial.value == min_expl_tune
    assert min_expl_tune < min_expl_orig


def test_tuner_finds_low_expl() -> None:
    """Tuner workflow is able to find parameters with low exploitability."""
    MAX_ITER = 600

    env = Environment.building_evacuation(n_floor=3, floor_l=5, floor_w=5)
    alg = OnlineMirrorDescent()

    optuna_study = alg.tune(
        metric=GeometricMean(shift=0),
        envs=[env],
        solve_kwargs={"max_iter": MAX_ITER, "atol": None, "rtol": None},
        n_trials=30,
    )
    assert optuna_study.best_trial.value is not None
    assert 0 <= optuna_study.best_trial.value <= 3e-4


def test_fictitious_play_save_and_load() -> None:
    fp = FictitiousPlay(alpha=0.5)
    with TemporaryDirectory() as tmpdir:
        fp.save(tmpdir)
        fp2 = FictitiousPlay.load(tmpdir)
        assert fp2.alpha == fp.alpha


@pytest.mark.parametrize("n_inner", [None, 50])
def test_prior_descent_save_and_load(n_inner: int | None) -> None:
    pd = PriorDescent(eta=0.5, n_inner=n_inner)
    with TemporaryDirectory() as tmpdir:
        pd.save(tmpdir)
        pd2 = PriorDescent.load(tmpdir)
        assert pd2.eta == pd.eta
        assert pd2.n_inner == pd.n_inner


def test_mirror_descent_save_and_load() -> None:
    md = OnlineMirrorDescent(alpha=2.0)
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
