from __future__ import annotations

import abc
import warnings
from typing import Any, Literal, TypeVar

import numpy as np
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
        pi: Literal["uniform"] | torch.Tensor = "uniform",
        max_iter: int,
        atol: float,
        rtol: float,
        metric: Literal["shifted_geo_mean", "failure_rate"],
        stat: Literal["iterations", "runtime", "exploitability"] = "iterations",
        fail_thresh: (
            int | float | None
        ) = None,  # used for metric="failure_rate" and final checking/warning
        shift: float | None = None,  # only used for metric="shifted_geo_mean"
        n_trials: int | None,
        timeout: float,
        drop_on_failure: bool = True,
        tuner_instance_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Optimize optuna study object."""
        # set fail_thresh and shift values
        if fail_thresh is None:
            if stat == "iterations":
                fail_thresh = max_iter
                print(
                    f"fail_thresh not specified; adopt fail_thresh = {max_iter=} for {stat=}"
                )
            elif stat == "exploitability":
                fail_thresh = atol if atol is not None else 1e-3
                print(
                    f"fail_thresh not specified; adopt fail_thresh = atol (rtol ignored) = {fail_thresh} for {stat=}"
                )
            else:
                raise ValueError(f"need to specify fail_thresh for {stat=}")

        shift = 10 if shift is None else shift

        def objective(trial: optuna.Trial) -> float:
            # consider passing variables explicitly;
            # otherwise can lead to weird errors if assign/overwrite the variables inside which could lead to variables referenced before assignment errors
            stats: list[int | float] = []
            if tuner_instance_kwargs is not None:
                solver = self._tuner_instance(trial, **tuner_instance_kwargs)
            else:
                solver = self._tuner_instance(trial)
            for env_instance in env_suite:
                solutions, expls, runtimes = solver.solve(
                    env_instance, pi=pi, max_iter=max_iter, atol=atol, rtol=rtol
                )
                if stat == "iterations":
                    stats.append(len(solutions) - 1)
                elif stat == "runtime":
                    stats.append(runtimes[-1])
                elif stat == "exploitability":
                    stats.append(np.min(expls))
                else:
                    raise ValueError(f"unexpected {stat=}")

            if metric == "failure_rate":
                return failure_rate(stats, fail_thresh=fail_thresh)
            elif metric == "shifted_geo_mean":
                return shifted_geometric_mean(stats, shift=shift)
            else:
                raise ValueError(f"unexpected {metric=}")

        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=0)
        )  # TODO: shall we consider any other sampler/optimizer for optuna?
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        fail_thresh_final = fail_thresh if metric == "shifted_geo_mean" else 1

        if study.best_trial.value and study.best_trial.value >= fail_thresh_final:
            warnings.warn(
                "None of the algorithm trials reached the given "
                "exploitability threshold within the specified number of "
                "iterations in any of the instances in the environment suite. "
            )
            if drop_on_failure:
                return None
            else:
                return study.best_trial.params
        else:
            return study.best_trial.params
