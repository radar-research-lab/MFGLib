from __future__ import annotations

import abc
import json
import time
import warnings
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Final,
    Generic,
    Iterable,
    Literal,
    Protocol,
    Sequence,
    TypedDict,
    TypeVar,
)

import optuna
import torch
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import PartialFixedSampler
from rich import box
from rich import print as rich_print
from rich.console import Group
from rich.panel import Panel
from rich.pretty import pretty_repr
from rich.table import Table
from rich.text import Text

from mfglib import __version__
from mfglib.alg.utils import _trigger_early_stopping
from mfglib.env import Environment
from mfglib.scoring import exploitability_score as expl_score

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
        env: Environment,
        *,
        pi_0: Literal["uniform"] | torch.Tensor = ...,
        max_iter: int = ...,
        atol: float | None = ...,
        rtol: float | None = ...,
        verbose: int = ...,
    ) -> tuple[list[torch.Tensor], list[float], list[float]]:
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

    def tune(
        self,
        metric: Metric,
        envs: Sequence[Environment],
        pi_0s: Sequence[torch.Tensor] | Literal["uniform"] = "uniform",
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
            pi_0s
                Policy initializations. ``envs`` and ``pi_0s`` are "zipped" together when
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
                if pi_0s == "uniform":
                    pi_0: Literal["uniform"] | torch.Tensor = "uniform"
                else:
                    pi_0 = pi_0s[i]
                pi, expl, rt = solver.solve(env, pi_0=pi_0, **solve_kwargs)
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


class State(Protocol):
    pi: torch.Tensor


T = TypeVar("T", bound=State)


class Iterative(Algorithm, Generic[T]):

    def solve(
        self,
        env: Environment,
        *,
        pi_0: Literal["uniform"] | torch.Tensor = "uniform",
        max_iter: int = DEFAULT_MAX_ITER,
        atol: float | None = DEFAULT_ATOL,
        rtol: float | None = DEFAULT_RTOL,
        verbose: int = 0,
    ) -> tuple[list[torch.Tensor], list[float], list[float]]:
        if pi_0 == "uniform":
            pi_0 = torch.ones((env.T + 1,) + env.S + env.A) / env.n_actions
        elif isinstance(pi_0, torch.Tensor):
            pi_0 = pi_0.detach().clone()
        else:
            raise TypeError("invalid pi_0 provided")

        pis = [pi_0]
        argmin = 0
        expls = [expl_score(env, pi_0)]
        rts = [0.0]

        state = self.init_state(env, pi_0)

        logger = Logger(verbose)
        logger.display_info(
            env=env,
            cls=f"{self.__class__.__name__}",
            parameters=self.parameters,
            atol=atol,
            rtol=rtol,
            max_iter=max_iter,
        )
        logger.insert_row(
            i=0,
            expl=expls[0],
            ratio=expls[0] / expls[0],
            argmin=argmin,
            elapsed=rts[0],
        )

        if _trigger_early_stopping(expls[0], expls[0], atol, rtol):
            logger.flush_stopped()
            return pis, expls, rts

        t_0 = time.time()
        for i in range(1, max_iter + 1):
            state = self.step_state(state)
            pis += [state.pi]
            expls += [expl_score(env, pis[i])]
            rts += [time.time() - t_0]
            if expls[i] < expls[argmin]:
                argmin = i
            logger.insert_row(
                i=i,
                expl=expls[i],
                ratio=expls[i] / expls[0],
                argmin=argmin,
                elapsed=rts[i],
            )
            if _trigger_early_stopping(expls[0], expls[i], atol, rtol):
                logger.flush_stopped()
                return pis, expls, rts

        logger.flush_exhausted()
        return pis, expls, rts

    @abc.abstractmethod
    def init_state(self, env: Environment, pi_0: torch.Tensor) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    def step_state(self, state: T) -> T:
        raise NotImplementedError

    @abc.abstractmethod
    @property
    def parameters(self) -> dict[str, float | str | None]:
        raise NotImplementedError


class Logger:
    INFO_PANEL_WIDTH: Final = 61
    MAX_TABLE_LENGTH: Final = 50

    def __init__(self, verbose: int) -> None:
        self.verbose = verbose
        self.table = self.create_empty_table()

    @staticmethod
    def create_empty_table() -> Table:
        return Table(
            "Iter (n)",
            "Expl_n",
            "Ratio_n",
            "Argmin_n",
            "Elapsed_n",
        )

    def display_info(
        self,
        env: Environment,
        cls: str,
        parameters: dict[str, float | str | None],
        atol: float | None,
        rtol: float | None,
        max_iter: int,
    ) -> None:
        if self.verbose:
            top_group = Group(
                Text(
                    f"MFGLib v{__version__} : A Library for Mean-Field Games",
                    justify="center",
                    style="bold",
                ),
                Text("RADAR Research Lab, UC Berkeley", justify="center"),
            )
            top_panel = Panel(
                top_group, box=box.HEAVY, width=self.INFO_PANEL_WIDTH, padding=1
            )

            env_group = Group(
                f"S = {env.S}",
                f"A = {env.A}",
                f"T = {env.T}",
                f"r_max = {env.r_max}",
            )
            env_panel = Panel(
                env_group,
                title=Text("Environment Summary", style="bold"),
                width=self.INFO_PANEL_WIDTH,
                title_align="left",
                box=box.SQUARE,
            )

            alg_group = Group(
                f"class = {cls}",
                f"parameters = {pretty_repr(parameters)}",
                f"atol = {atol}",
                f"rtol = {rtol}",
                f"max_iter = {max_iter}",
            )
            alg_panel = Panel(
                alg_group,
                title=Text("Algorithm Summary", style="bold"),
                width=self.INFO_PANEL_WIDTH,
                title_align="left",
                box=box.SQUARE,
            )

            doc_group = Group(
                f"- The verbosity level is set to {self.verbose}.",
                "- Table output is printed 50 rows at a time.",
                "- Ratio_n := Expl_n / Expl_0.",
                "- Argmin_n := Argmin_{0≤i≤n} Expl_i.",
                "- Elapsed_n measures time in seconds.",
            )
            doc_panel = Panel(
                doc_group,
                title=Text("Documentation", style="bold"),
                width=self.INFO_PANEL_WIDTH,
                title_align="left",
                box=box.SQUARE,
            )
            rich_print(top_panel, env_panel, alg_panel, doc_panel)

    def insert_row(
        self, i: int, expl: float, ratio: float, argmin: int, elapsed: float
    ) -> None:
        if self.verbose and i % self.verbose == 0:
            self.table.add_row(
                f"{i}",
                f"{expl:.4e}",
                f"{ratio:.4e}",
                f"{argmin}",
                f"{elapsed:.2e}",
            )
            if len(self.table.rows) == self.MAX_TABLE_LENGTH:
                rich_print(self.table)
                self.table = self.create_empty_table()

    def flush_exhausted(self) -> None:
        if self.verbose:
            rich_print(self.table)
            rich_print("Number of iterations exhausted.")

    def flush_stopped(self) -> None:
        if self.verbose:
            rich_print(self.table)
            rich_print("Absolute or relative stopping criteria met.")
