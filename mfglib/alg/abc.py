from __future__ import annotations

import abc
import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Final, Iterable, Literal, Sequence, TypedDict, TypeVar

import optuna
import torch
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import PartialFixedSampler

from mfglib.env import Environment

if TYPE_CHECKING:
    from mfglib.tuning import Metric

Self = TypeVar("Self", bound="Algorithm")

DEFAULT_MAX_ITER: Final = 100
DEFAULT_ATOL: Final = 1e-3
DEFAULT_RTOL: Final = 1e-3


class SolveKwargs(TypedDict, total=False):
    max_iter: int
    atol: float | None
    rtol: float | None
    verbose: bool


class Algorithm(abc.ABC):
    """Abstract interface for all algorithms."""

    @abc.abstractmethod
    def solve(
        self,
        env_instance: Environment,
        *,
        pi: Literal["uniform"] | torch.Tensor = "uniform",
        max_iter: int = DEFAULT_MAX_ITER,
        atol: float | None = DEFAULT_ATOL,
        rtol: float | None = DEFAULT_RTOL,
        verbose: int = 0,
    ) -> tuple[list[torch.Tensor], list[float], list[float]]:
        """Run the algorithm and solve for a Nash-Equilibrium policy."""
        raise NotImplementedError

    def save(self, path: Path | str) -> None:
        path = Path(path) / self.__class__.__name__
        path.mkdir(exist_ok=False)
        with open(path / "kwargs.json", "w") as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def load(cls: type[Self], path: Path | str) -> Self:
        path = Path(path) / cls.__name__
        with open(path / "kwargs.json", "r") as f:
            kwargs = json.load(f)
        return cls(**kwargs)

    @abc.abstractmethod
    def __str__(self) -> str:
        """Represent algorithm instance and associated parameters with a string."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def _init_tuner_instance(cls: type[Self], trial: optuna.Trial) -> Self:
        raise NotImplementedError

    @classmethod
    def from_study(cls: type[Self], study: optuna.Study) -> Self:
        return cls(**study.best_params)

    def tune(
        self,
        metric: Metric,
        envs: Sequence[Environment],
        pi0s: Sequence[torch.Tensor] | Literal["uniform"] = "uniform",
        solve_kwargs: SolveKwargs | None = None,
        sampler: optuna.samplers.BaseSampler | None = None,
        frozen_attrs: Iterable[str] | None = None,
        n_trials: int | None = None,
        timeout: float | None = None,
    ) -> optuna.Study:
        """Tune the algorithm over multiple environment/initialization pairs.

        Args
        ----
            metric
                Objective function to minimizer.
            envs
                List of environment targets.
            pi0s
                Policy initializations. ``envs`` and ``pi0s`` are "zipped" together when
                computing the metrics.
            solve_kwargs
                Additional keyword arguments passed to the solver.
            sampler
                The sampler used to explore the search space of the optimization.
                If ``None``, the default sampler ``optuna.samplers.TPESampler`` is
                used. The sampler guides how different hyperparameter trials are
                selected.
            frozen_attrs
                A list of attributes that should be frozen (i.e., fixed) during the
                optimization process. These attributes will not be considered for
                optimization, and their values will be taken directly from the instance
                of the class.
            n_trials
                The  number of trials to run. Refer to ``optuna`` documentation for
                further details on the handling of ``None``.
            timeout
                Stop study after the given number of second(s). Refer to ``optuna``
                documentation for further details.

        Returns
        -------
        optuna.Study
            The result of the hyperparameter tuning process.

        """
        if sampler is None:
            sampler = optuna.samplers.TPESampler()

        solve_kwargs = solve_kwargs or {}

        fixed_params = {}
        for attr in frozen_attrs or []:
            if hasattr(self, attr):
                fixed_params[attr] = getattr(self, attr)

        def objective(trial: optuna.Trial) -> float:
            solver = self._init_tuner_instance(trial)
            pis, expls, rts = [], [], []
            for i, env in enumerate(envs):
                if pi0s == "uniform":
                    pi0: Literal["uniform"] | torch.Tensor = "uniform"
                else:
                    pi0 = pi0s[i]
                pi, expl, rt = solver.solve(env, pi=pi0, **solve_kwargs)
                pis += [pi]
                expls += [expl]
                rts += [rt]
            return metric.evaluate(pis, expls, rts, solve_kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ExperimentalWarning)
            study = optuna.create_study(
                sampler=PartialFixedSampler(fixed_params, sampler)
            )
            study.optimize(objective, n_trials=n_trials, timeout=timeout)

        return study
