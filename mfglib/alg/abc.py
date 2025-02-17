from __future__ import annotations

import abc
import json
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, TypeVar

import numpy as np
import optuna
import torch
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import PartialFixedSampler

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

    def tune_on_failure_rate(
        self,
        envs: Iterable[Environment],
        *,
        pi: Literal["uniform"] | torch.Tensor = "uniform",
        stat: Literal["iter", "rt", "expl"] = "iter",
        fail_thresh: float | None = None,
        max_iter: int = 100,
        atol: float = 1e-3,
        rtol: float = 1e-3,
        sampler: optuna.samplers.BaseSampler | None = None,
        frozenattrs: Iterable[str] | None = None,
        n_trials: int | None = None,
        timeout: int | None = None,
        verbose: bool = True,
    ) -> optuna.Study:
        """Tune the algorithm over a given environment suite.

        The objective is to minimize the failure rate across the
        provided environments.

        Args
        ----
        envs
            A list of environment instances to optimize the solver over.
        pi
            The policy with which the solver is initialized.
        stat
            The statistic to monitor during optimization. It can be one
            of the following:
            - "iter": the number of iterations taken by the solver.
            - "rt": the runtime (in seconds) taken by the solver.
            - "expl": the minimum exploitability score observed during
                the solution process.
        fail_thresh
            The failure threshold used to identify if the solver 'failed' on
            a particular environment instance. This threshold is compared
            against the chosen statistic (`stat`). For instance, if `stat="iter"`,
            it defines the maximum number of iterations the solver is allowed to
            run before considering it to have failed. If `None`, the default
            threshold is determined based on the value of `stat`.
        max_iter
            The maximum number of iterations to run the algorithm on each
            environment instance.
        atol
            Absolute tolerance criteria for early stopping of the solver.
        rtol
            Relative tolerance criteria for early stopping of the solver.
        sampler
            The sampler used to explore the search space of the optimization.
            If `None`, the default sampler `optuna.samplers.TPESampler` is used.
            The sampler guides how different hyperparameter trials are selected.
        frozenattrs
            A list of attributes that should be frozen (i.e., fixed) during the
            optimization process. These attributes will not be considered for
            optimization, and their values will be taken directly from the instance
            of the class.
        n_trials
            The number of trials. If this argument is not given, as many
            trials are run as possible.
        timeout
            Stop tuning after the given number of second(s) on each
            environment instance. If this argument is not given, as many trials are
            run as possible.
        verbose
            Whether to print additional information during the optimization process.
            If `True`, the function will print debugging messages and progress updates.
        """
        if fail_thresh is None:
            if stat == "iter":
                if verbose:
                    print(f"adopting fail_thresh = max_iter = {max_iter} for {stat=}")
                fail_thresh = max_iter
            elif stat == "rt":
                raise ValueError(f"must specify fail_thresh for {stat=}")
            elif stat == "expl":
                if verbose:
                    print(
                        "adopting fail_thresh = atol (rtol ignored) "
                        f"= {atol} for {stat=}"
                    )
                fail_thresh = atol
            else:
                raise ValueError("stat must be 'iter' or 'rt' or 'expl'")

        def objective(trial: optuna.Trial) -> float:
            solver = self._init_tuner_instance(trial)
            stats = self._collect_stats(stat, envs, solver, pi, max_iter, atol, rtol)
            failure_rate: float = (stats >= fail_thresh).mean()
            return failure_rate

        return self._optimize_study(objective, sampler, frozenattrs, n_trials, timeout)

    def tune_on_geometric_mean(
        self,
        envs: Iterable[Environment],
        *,
        pi: Literal["uniform"] | torch.Tensor = "uniform",
        stat: Literal["iter", "rt", "expl"] = "iter",
        shift: float = 0,
        max_iter: int = 100,
        atol: float = 1e-3,
        rtol: float = 1e-3,
        sampler: optuna.samplers.BaseSampler | None = None,
        frozenattrs: Iterable[str] | None = None,
        n_trials: int | None = None,
        timeout: int | None = None,
    ) -> optuna.Study:
        """Tune the algorithm over a given environment suite.

        The objective is to minimize the (shifted) geometric mean across the
        provided environments. A "shifted geometric mean" is a variation of the
        standard geometric mean where each data point is first shifted by adding
        a constant value (the "shift") before calculating the geometric mean;
        essentially, it mitigates the impact of extreme outliers by adjusting the
        data before computing the average, making it more robust to skewed
        distributions.

        Args
        ----
        envs
            A list of environment instances to optimize the solver over.
        pi
            The policy with which the solver is initialized.
        stat
            The statistic to monitor during optimization. It can be one
            of the following:
            - "iter": the number of iterations taken by the solver.
            - "rt": the runtime (in seconds) taken by the solver.
            - "expl": the minimum exploitability score observed during
                the solution process.
        shift
            An additional shift value for the geometric mean. Defaults to zero.
        max_iter
            The maximum number of iterations to run the algorithm on each
            environment instance.
        atol
            Absolute tolerance criteria for early stopping of the solver.
        rtol
            Relative tolerance criteria for early stopping of the solver.
        sampler
            The sampler used to explore the search space of the optimization.
            If `None`, the default sampler `optuna.samplers.TPESampler` is used.
            The sampler guides how different hyperparameter trials are selected.
        frozenattrs
            A list of attributes that should be frozen (i.e., fixed) during the
            optimization process. These attributes will not be considered for
            optimization, and their values will be taken directly from the instance
            of the class.
        n_trials
            The number of trials. If this argument is not given, as many
            trials are run as possible.
        timeout
            Stop tuning after the given number of second(s) on each
            environment instance. If this argument is not given, as many trials are
            run as possible.
        """
        if stat not in ["iter", "rt", "expl"]:
            raise ValueError("stat must be 'iter' or 'rt' or 'expl'")

        def objective(trial: optuna.Trial) -> float:
            solver = self._init_tuner_instance(trial)
            stats = self._collect_stats(stat, envs, solver, pi, max_iter, atol, rtol)
            geomean: float = ((stats + shift).prod() ** (1.0 / stats.size)) - shift
            return geomean

        return self._optimize_study(objective, sampler, frozenattrs, n_trials, timeout)

    @staticmethod
    def _collect_stats(
        stat: Literal["iter", "rt", "expl"],
        envs: Iterable[Environment],
        solver: Algorithm,
        pi: Literal["uniform"] | torch.Tensor,
        max_iter: int,
        atol: float,
        rtol: float,
    ) -> np.ndarray[Any, Any]:
        stats: list[float] = []
        for env in envs:
            solns, expls, rts = solver.solve(
                env, pi=pi, max_iter=max_iter, atol=atol, rtol=rtol
            )
            if stat == "iter":
                stats.append(len(solns) - 1)
            elif stat == "rt":
                stats.append(rts[-1])
            elif stat == "expl":
                stats.append(min(expls))
        return np.array(stats)

    def _optimize_study(
        self,
        objective: Callable[[optuna.Trial], float],
        sampler: optuna.samplers.BaseSampler | None,
        frozenattrs: Iterable[str] | None,
        n_trials: int | None,
        timeout: int | None,
    ) -> optuna.Study:
        if sampler is None:
            sampler = optuna.samplers.TPESampler()

        fixed_params = {}
        for attr in frozenattrs or []:
            value = getattr(self, attr)
            if value is not None:
                fixed_params[attr] = value

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ExperimentalWarning)
            study = optuna.create_study(
                sampler=PartialFixedSampler(fixed_params, sampler)
            )
            study.optimize(objective, n_trials=n_trials, timeout=timeout)

        return study
