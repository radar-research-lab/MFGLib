from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Literal, Protocol, cast

import numpy as np

if TYPE_CHECKING:
    import torch

    from mfglib.alg.abc import DEFAULT_ATOL, DEFAULT_MAX_ITER, Algorithm, SolveKwargs
    from mfglib.env import Environment


class Metric(Protocol):
    def score(
        self,
        solver: Algorithm,
        envs: Iterable[Environment],
        pi0s: Iterable[torch.Tensor],
        solver_kwargs: SolveKwargs,
    ) -> float: ...


class FailureRate:
    def __init__(
        self,
        fail_thresh: float | int | None = None,
        stat: Literal["iter", "rt", "expl"] = "expl",
    ) -> None:
        """Failure rate metric.

        Args:
            fail_thresh: The failure threshold used to identify if the solver 'failed'
                on a particular environment instance. This threshold is compared
                against the chosen statistic (`stat`). For instance, if `stat="iter"`,
                it defines the maximum number of iterations the solver is allowed to
                run before considering it to have failed. If `None`, the default
                threshold is determined based on the value of `stat`.
            stat: The statistic to monitor during optimization. It can be one
                of the following:
                - "iter": the number of iterations taken by the solver.
                - "rt": the runtime (in seconds) taken by the solver.
                - "expl": the minimum exploitability score observed during
                    the solution process.
        """
        if stat not in ("iter", "rt", "expl"):
            raise ValueError("stat must be 'iter' or 'rt' or 'expl'")
        self.stat = stat
        self.fail_thresh = fail_thresh

    def score(
        self,
        solver: Algorithm,
        envs: Iterable[Environment],
        pi0s: Iterable[torch.Tensor],
        solve_kwargs: SolveKwargs,
    ) -> float:
        if self.fail_thresh is None:
            if self.stat == "iter":
                fail_thresh = float(solve_kwargs.get("max_iter", DEFAULT_MAX_ITER))
            elif self.stat == "expl":
                atol = solve_kwargs.get("atol", DEFAULT_ATOL)
                if atol is None:
                    raise ValueError(
                        f"must specify fail_thresh for {self.stat=} when {atol=}"
                    )
                fail_thresh = atol
            else:
                raise ValueError(f"must specify fail_thresh for {self.stat=}")
        else:
            fail_thresh = self.fail_thresh

        stats = []
        for env in envs:
            for pi0 in pi0s:
                solns, expls, rts = solver.solve(env, pi=pi0, **solve_kwargs)
                if self.stat == "iter":
                    stats.append(len(solns) - 1.0)
                elif self.stat == "rt":
                    stats.append(rts[-1])
                elif self.stat == "expl":
                    stats.append(min(expls))
        stats_arr = np.array(stats)
        return cast(float, (stats_arr >= fail_thresh).mean())


class GeometricMean:
    def __init__(
        self, shift: float = 0, stat: Literal["iter", "rt", "expl"] = "expl"
    ) -> None:
        """Geometric mean metric.

        Args:
            shift: An additional shift value for the geometric mean. Defaults to zero.
            stat: The statistic to monitor during optimization. It can be one
                of the following:
                - "iter": the number of iterations taken by the solver.
                - "rt": the runtime (in seconds) taken by the solver.
                - "expl": the minimum exploitability score observed during
                    the solution process.
        """
        if stat not in ("iter", "rt", "expl"):
            raise ValueError("stat must be 'iter' or 'rt' or 'expl'")
        self.stat = stat
        self.shift = shift

    def score(
        self,
        solver: Algorithm,
        envs: Iterable[Environment],
        pi0s: Iterable[torch.Tensor],
        solve_kwargs: SolveKwargs,
    ) -> float:
        stats = []
        for env in envs:
            for pi0 in pi0s:
                solns, expls, rts = solver.solve(env, pi=pi0, **solve_kwargs)
                if self.stat == "iter":
                    stats.append(len(solns) - 1.0)
                elif self.stat == "rt":
                    stats.append(rts[-1])
                elif self.stat == "expl":
                    stats.append(min(expls))
        stats_arr = np.array(stats)
        return cast(
            float,
            ((stats_arr + self.shift).prod() ** (1.0 / stats_arr.size)) - self.shift,
        )
