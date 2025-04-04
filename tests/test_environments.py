import numpy as np
import pytest
import torch

from mfglib.alg import OnlineMirrorDescent
from mfglib.env import Environment


@pytest.mark.parametrize("T", [2, 4])
@pytest.mark.parametrize("n", [4, 10])
@pytest.mark.parametrize("p_still", [0.1, 0.9])
def test_beach_bar(T: int, n: int, p_still: float) -> None:
    env = Environment.beach_bar(T=T, n=n, p_still=p_still)

    size_tsa = (T + 1, n, 3)
    size_ssa = (n, n, 3)
    size_ts = (T + 1, n)
    size_sa = (n, 3)
    l_s = 1
    l_a = 1

    L = torch.ones(size=size_tsa) / np.prod(size_sa).item()
    r = env.reward(0, L[0])
    p = env.prob(0, L[0])
    assert r.shape == size_sa
    assert p.shape == size_ssa
    assert (p >= 0).all()
    assert torch.allclose(
        p.sum(dim=tuple(range(l_s))), torch.ones(size=size_sa), rtol=0, atol=1e-5
    )

    algorithm = OnlineMirrorDescent(alpha=0.1)
    solns, _, _ = algorithm.solve(env, max_iter=100, atol=None, rtol=None)
    assert solns[-1].shape == size_tsa
    assert (solns[-1] >= 0).all()
    assert torch.allclose(
        solns[-1].sum(dim=tuple(range(-l_a, 0))),
        torch.ones(size=size_ts),
        rtol=0,
        atol=1e-5,
    )


@pytest.mark.parametrize("T", [2, 4])
@pytest.mark.parametrize("n_floor", [3, 6])
@pytest.mark.parametrize("floor_l", [2, 4])
@pytest.mark.parametrize("floor_w", [2, 4])
def test_building_evacuation(T: int, n_floor: int, floor_l: int, floor_w: int) -> None:
    env = Environment.building_evacuation(
        T=T, n_floor=n_floor, floor_l=floor_l, floor_w=floor_w
    )

    size_tsa = (T + 1, n_floor, floor_l, floor_w, 6)
    size_ssa = (n_floor, floor_l, floor_w, n_floor, floor_l, floor_w, 6)
    size_ts = (T + 1, n_floor, floor_l, floor_w)
    size_sa = (n_floor, floor_l, floor_w, 6)
    l_s = 3
    l_a = 1

    L = torch.ones(size=size_tsa) / np.prod(size_sa).item()
    r = env.reward(0, L[0])
    p = env.prob(0, L[0])
    assert r.shape == size_sa
    assert p.shape == size_ssa
    assert (p >= 0).all()
    assert torch.allclose(
        p.sum(dim=tuple(range(l_s))), torch.ones(size=size_sa), rtol=0, atol=1e-5
    )

    algorithm = OnlineMirrorDescent(alpha=0.1)
    solns, _, _ = algorithm.solve(env, max_iter=100, atol=None, rtol=None)
    assert solns[-1].shape == size_tsa
    assert (solns[-1] >= 0).all()
    assert torch.allclose(
        solns[-1].sum(dim=tuple(range(-l_a, 0))),
        torch.ones(size=size_ts),
        rtol=0,
        atol=1e-5,
    )


@pytest.mark.parametrize("T", [2, 4])
@pytest.mark.parametrize("n", [3, 6])
def test_conservative_treasure_hunting(T: int, n: int) -> None:
    env = Environment.conservative_treasure_hunting(T=T, n=n, r=(1,) * n, c=(1,) * T)

    size_tsa = (T + 1, n, n)
    size_ssa = (n, n, n)
    size_ts = (T + 1, n)
    size_sa = (n, n)
    l_s = 1
    l_a = 1

    L = torch.ones(size=size_tsa) / np.prod(size_sa).item()
    r = env.reward(0, L[0])
    p = env.prob(0, L[0])
    assert r.shape == size_sa
    assert p.shape == size_ssa
    assert (p >= 0).all()
    assert torch.allclose(
        p.sum(dim=tuple(range(l_s))), torch.ones(size=size_sa), rtol=0, atol=1e-5
    )

    algorithm = OnlineMirrorDescent(alpha=0.1)
    solns, _, _ = algorithm.solve(env, max_iter=100, atol=None, rtol=None)
    assert solns[-1].shape == size_tsa
    assert (solns[-1] >= 0).all()
    assert torch.allclose(
        solns[-1].sum(dim=tuple(range(-l_a, 0))),
        torch.ones(size=size_ts),
        rtol=0,
        atol=1e-5,
    )


@pytest.mark.parametrize("T", [3, 5])
@pytest.mark.parametrize("torus_l", [2, 4])
@pytest.mark.parametrize("torus_w", [2, 4])
@pytest.mark.parametrize("p_still", [0.1, 0.9])
def test_crowd_motion(T: int, torus_l: int, torus_w: int, p_still: float) -> None:
    env = Environment.crowd_motion(
        T=T, torus_l=torus_l, torus_w=torus_w, p_still=p_still
    )

    size_tsa = (T + 1, torus_l, torus_w, 5)
    size_ssa = (torus_l, torus_w, torus_l, torus_w, 5)
    size_ts = (T + 1, torus_l, torus_w)
    size_sa = (torus_l, torus_w, 5)
    l_s = 2
    l_a = 1

    L = torch.ones(size=size_tsa) / np.prod(size_sa).item()
    r = env.reward(0, L[0])
    p = env.prob(0, L[0])
    assert r.shape == size_sa
    assert p.shape == size_ssa
    assert (p >= 0).all()
    assert torch.allclose(
        p.sum(dim=tuple(range(l_s))), torch.ones(size=size_sa), rtol=0, atol=1e-5
    )

    algorithm = OnlineMirrorDescent(alpha=0.1)
    solns, _, _ = algorithm.solve(env, max_iter=100, atol=None, rtol=None)
    assert solns[-1].shape == size_tsa
    assert (solns[-1] >= 0).all()
    assert torch.allclose(
        solns[-1].sum(dim=tuple(range(-l_a, 0))),
        torch.ones(size=size_ts),
        rtol=0,
        atol=1e-5,
    )


@pytest.mark.parametrize("T", [3, 5])
@pytest.mark.parametrize("s_inv", [4, 6])
@pytest.mark.parametrize("Q", [1, 3])
@pytest.mark.parametrize("H", [1, 3])
@pytest.mark.parametrize("sigma", [1, 2])
def test_equilibrium_price(T: int, s_inv: int, Q: int, H: int, sigma: float) -> None:
    env = Environment.equilibrium_price(T=T, s_inv=s_inv, Q=Q, H=H, sigma=sigma)

    size_tsa = (T + 1, s_inv + 1, Q + 1, H + 1)
    size_ssa = (s_inv + 1, s_inv + 1, Q + 1, H + 1)
    size_ts = (T + 1, s_inv + 1)
    size_sa = (s_inv + 1, Q + 1, H + 1)
    l_s = 1
    l_a = 2

    L = torch.ones(size=size_tsa) / np.prod(size_sa).item()
    r = env.reward(0, L[0])
    p = env.prob(0, L[0])
    assert r.shape == size_sa
    assert p.shape == size_ssa
    assert (p >= 0).all()
    assert torch.allclose(
        p.sum(dim=tuple(range(l_s))), torch.ones(size=size_sa), rtol=0, atol=1e-5
    )

    algorithm = OnlineMirrorDescent(alpha=0.1)
    solns, _, _ = algorithm.solve(env, max_iter=100, atol=None, rtol=None)
    assert solns[-1].shape == size_tsa
    assert (solns[-1] >= 0).all()
    assert torch.allclose(
        solns[-1].sum(dim=tuple(range(-l_a, 0))),
        torch.ones(size=size_ts),
        rtol=0,
        atol=1e-5,
    )


def test_left_right() -> None:
    env = Environment.left_right()

    size_tsa = (2, 3, 2)
    size_ssa = (3, 3, 2)
    size_ts = (2, 3)
    size_sa = (3, 2)
    l_s = 1
    l_a = 1

    L = torch.ones(size=size_tsa) / np.prod(size_sa).item()
    r = env.reward(0, L[0])
    p = env.prob(0, L[0])
    assert r.shape == size_sa
    assert p.shape == size_ssa
    assert (p >= 0).all()
    assert torch.allclose(
        p.sum(dim=tuple(range(l_s))), torch.ones(size=size_sa), rtol=0, atol=1e-5
    )

    algorithm = OnlineMirrorDescent(alpha=0.1)
    solns, _, _ = algorithm.solve(env, max_iter=100, atol=None, rtol=None)
    assert solns[-1].shape == size_tsa
    assert (solns[-1] >= 0).all()
    assert torch.allclose(
        solns[-1].sum(dim=tuple(range(-l_a, 0))),
        torch.ones(size=size_ts),
        rtol=0,
        atol=1e-5,
    )


def test_linear_quadratic() -> None:
    env = Environment.linear_quadratic(T=2, el=2, m=1)

    size_tsa = (3, 5, 3)
    size_ssa = (5, 5, 3)
    size_ts = (3, 5)
    size_sa = (5, 3)
    l_s = 1
    l_a = 1

    L = torch.ones(size=size_tsa) / np.prod(size_sa).item()
    r = env.reward(0, L[0])
    p = env.prob(0, L[0])
    assert r.shape == size_sa
    assert p.shape == size_ssa
    assert (p >= 0).all()
    assert torch.allclose(
        p.sum(dim=tuple(range(l_s))), torch.ones(size=size_sa), rtol=0, atol=1e-5
    )

    algorithm = OnlineMirrorDescent(alpha=0.1)
    solns, _, _ = algorithm.solve(env, max_iter=100, atol=None, rtol=None)
    assert solns[-1].shape == size_tsa
    assert (solns[-1] >= 0).all()
    assert torch.allclose(
        solns[-1].sum(dim=tuple(range(-l_a, 0))),
        torch.ones(size=size_ts),
        rtol=0,
        atol=1e-5,
    )


@pytest.mark.parametrize("T", [2, 4])
@pytest.mark.parametrize("n", [3, 6])
@pytest.mark.parametrize("m", [1.0, 10.0])
def test_random_linear(T: int, n: int, m: float) -> None:
    env = Environment.random_linear(T=T, n=n, m=m)

    size_tsa = (T + 1, n, n)
    size_ssa = (n, n, n)
    size_ts = (T + 1, n)
    size_sa = (n, n)
    l_s = 1
    l_a = 1

    L = torch.ones(size=size_tsa) / np.prod(size_sa).item()
    r = env.reward(0, L[0])
    p = env.prob(0, L[0])
    assert r.shape == size_sa
    assert p.shape == size_ssa
    assert (p >= 0).all()
    assert torch.allclose(
        p.sum(dim=tuple(range(l_s))), torch.ones(size=size_sa), rtol=0, atol=1e-5
    )

    algorithm = OnlineMirrorDescent(alpha=0.1)
    solns, _, _ = algorithm.solve(env, max_iter=100, atol=None, rtol=None)
    assert solns[-1].shape == size_tsa
    assert (solns[-1] >= 0).all()
    assert torch.allclose(
        solns[-1].sum(dim=tuple(range(-l_a, 0))),
        torch.ones(size=size_ts),
        rtol=0,
        atol=1e-5,
    )


@pytest.mark.parametrize("T", [1, 5])
def test_rock_paper_scissors(T: int) -> None:
    env = Environment.rock_paper_scissors(T=T)

    size_tsa = (T + 1, 4, 3)
    size_ssa = (4, 4, 3)
    size_ts = (T + 1, 4)
    size_sa = (4, 3)
    l_s = 1
    l_a = 1

    L = torch.ones(size=size_tsa) / np.prod(size_sa).item()
    r = env.reward(0, L[0])
    p = env.prob(0, L[0])
    assert r.shape == size_sa
    assert p.shape == size_ssa
    assert (p >= 0).all()
    assert torch.allclose(
        p.sum(dim=tuple(range(l_s))), torch.ones(size=size_sa), rtol=0, atol=1e-5
    )

    algorithm = OnlineMirrorDescent(alpha=0.1)
    solns, _, _ = algorithm.solve(env, max_iter=100, atol=None, rtol=None)
    assert solns[-1].shape == size_tsa
    assert (solns[-1] >= 0).all()
    assert torch.allclose(
        solns[-1].sum(dim=tuple(range(-l_a, 0))),
        torch.ones(size=size_ts),
        rtol=0,
        atol=1e-5,
    )


@pytest.mark.parametrize("T", [10, 20])
def test_susceptible_infected(T: int) -> None:
    env = Environment.susceptible_infected(T=T)

    size_tsa = (T + 1, 2, 2)
    size_ssa = (2, 2, 2)
    size_ts = (T + 1, 2)
    size_sa = (2, 2)
    l_s = 1
    l_a = 1

    L = torch.ones(size=size_tsa) / np.prod(size_sa).item()
    r = env.reward(0, L[0])
    p = env.prob(0, L[0])
    assert r.shape == size_sa
    assert p.shape == size_ssa
    assert (p >= 0).all()
    assert torch.allclose(
        p.sum(dim=tuple(range(l_s))), torch.ones(size=size_sa), rtol=0, atol=1e-5
    )

    algorithm = OnlineMirrorDescent(alpha=0.1)
    solns, _, _ = algorithm.solve(env, max_iter=100, atol=None, rtol=None)
    assert solns[-1].shape == size_tsa
    assert (solns[-1] >= 0).all()
    assert torch.allclose(
        solns[-1].sum(dim=tuple(range(-l_a, 0))),
        torch.ones(size=size_ts),
        rtol=0,
        atol=1e-5,
    )
