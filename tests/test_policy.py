import pytest
from torch.testing import assert_close

from mfglib.env import Environment
from mfglib.policy import Policy


def test_policy() -> None:
    lr = Environment.left_right()

    pr = [
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
    ]

    handcrafted = Policy([pr, pr])
    stationary = Policy.stationary(pr)
    automatic = Policy.uniform()

    assert_close(handcrafted.build(lr), stationary.build(lr))
    assert_close(stationary.build(lr), automatic.build(lr))

    # test mismatched dimensions
    with pytest.raises(ValueError):
        Policy.stationary([[0.5, 0.5]]).build(lr)

    # test invalid probability distribution
    with pytest.raises(ValueError):
        Policy.stationary([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]).build(lr)
