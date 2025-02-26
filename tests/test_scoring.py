import torch

from mfglib.env import Environment
from mfglib.scoring import exploitability_score


def test_exploitability_score() -> None:
    lr = Environment.left_right()

    go_left_policy = torch.tensor(
        [
            [[1.0, 0.0], [0.5, 0.5], [0.5, 0.5]],
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        ]
    )
    assert exploitability_score(lr, go_left_policy) == 1.0

    go_right_policy = torch.tensor(
        [
            [[0.0, 1.0], [0.5, 0.5], [0.5, 0.5]],
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        ]
    )
    assert exploitability_score(lr, go_right_policy) == 2.0

    policies = [go_left_policy, go_right_policy]
    assert exploitability_score(lr, policies) == [1.0, 2.0]
