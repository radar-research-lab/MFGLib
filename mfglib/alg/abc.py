from __future__ import annotations

import abc
import warnings
from typing import Any, Literal, TypeVar

import optuna
import torch

from mfglib.alg.utils import failure_rate, shifted_geometric_mean
from mfglib.env import Environment

Self = TypeVar("Self", bound="Algorithm")


class Algorithm(abc.ABC):
    """Abstract interface for all algorithms."""

    @abc.abstractmethod
    def solve(
        self,
        env_instance: Environment,
        *,
        pi: Literal["uniform"] | torch.Tensor = "uniform",
        max_iter: int = 100,
        atol: float | None = 1e-3,
        rtol: float | None = 1e-3,
        verbose: bool = False,
    ) -> tuple[list[torch.Tensor], list[float], list[float]]:
        """Run the algorithm and solve for a Nash-Equilibrium policy."""
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self) -> str:
        """Represent algorithm instance and associated parameters with a string."""
        raise NotImplementedError

    @abc.abstractmethod
    def tune(
        self: Self,
        env_suite: list[Environment],
        *,
        max_iter: int,
        atol: float,
        rtol: float,
        metric: Literal["shifted_geo_mean", "failure_rate"],
        n_trials: int | None,
        timeout: float,
    ) -> Self:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def _tuner_instance(cls: type[Self], trial: optuna.Trial) -> Self:
        raise NotImplementedError

    def _optimize_optuna_study(
        self,
        *,
        env_suite: list[Environment],
        pi: Literal["uniform"] | torch.Tensor, 
        max_iter: int,
        atol: float,
        rtol: float,
        metric: Literal["shifted_geo_mean", "failure_rate"],
        n_trials: int | None,
        timeout: float,
    ) -> dict[str, Any] | None:
        """Optimize optuna study object."""

        def objective(trial: optuna.Trial) -> float:
            stats = []
            solver = self._tuner_instance(trial)
            for env_instance in env_suite:
                solutions, _, _ = solver.solve(
                    env_instance, pi=pi, max_iter=max_iter, atol=atol, rtol=rtol
                )
                stats.append(len(solutions) - 1)

            if metric == "failure_rate":
                return failure_rate(stats, fail_thresh=max_iter)
            elif metric == "shifted_geo_mean":
                return shifted_geometric_mean(stats, shift=10)
            else:
                raise ValueError(f"unexpected metric={metric}")

        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        fail_thresh = max_iter if metric == "shifted_geo_mean" else 1

        if study.best_trial.value and study.best_trial.value >= fail_thresh:
            warnings.warn(
                "None of the algorithm trials reached the given "
                "exploitability threshold within the specified number of "
                "iterations in any of the instances in the environment suite. "
            )
            return None
        else:
            return study.best_trial.params
