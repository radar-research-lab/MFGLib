from __future__ import annotations

import time
from typing import Any, Literal

import optuna
import torch

from mfglib.alg.abc import Algorithm
from mfglib.alg.mf_omo_constraints import mf_omo_constraints
from mfglib.alg.mf_omo_obj import mf_omo_obj
from mfglib.alg.mf_omo_policy_given_mean_field import mf_omo_policy
from mfglib.alg.mf_omo_residual_balancing import mf_omo_residual_balancing
from mfglib.alg.utils import (
    _ensure_free_tensor,
    _print_fancy_header,
    _print_fancy_table_row,
    _print_solve_complete,
    _trigger_early_stopping,
    hat_initialization,
    tuple_prod,
)
from mfglib.env import Environment
from mfglib.metrics import exploitability_score

DEFAULT_OPTIMIZER = {
    "name": "Adam",
    "config": {
        "lr": 0.1,
    },
}


def _verify(
    x: torch.Tensor | None, y: torch.Tensor | None, *, parameterize: bool
) -> torch.Tensor | None:
    """Verify that at most one of the two initialization values is provided.

    Args
    ----
    x
        The argument corresponding with `parameterize=False`.
    y
        The argument corresponding with `parameterize=True`.
        parameterize: Whether to return `x` or `y`.
    """
    if parameterize:
        if x is not None:
            raise ValueError(
                "got an unexpected parameter for `parameterize=True`; "
                "the valid initialization arguments are `u`, `v`, and `w`"
            )
        return y
    else:
        if y is not None:
            raise ValueError(
                "got an unexpected parameter for `parameterize=False`; "
                "the valid initialization arguments are `L`, `z`, and `y`"
            )
        return x


class MFOMO(Algorithm):
    """Mean-Field Occupation Measure Optimization algorithm.

    Notes
    -----
    See [#mf1]_ for algorithm details.

    .. [#mf1] MF-OMO: An Optimization Formulation of Mean-Field Games
        Guo, X., Hu, A., & Zhang, J. (2022). arXiv:2206.09608.
    """

    def __init__(
        self,
        L: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        u: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        w: torch.Tensor | None = None,
        loss: Literal["l1"] | Literal["l2"] | Literal["l1_l2"] = "l1_l2",
        c1: float = 1.0,
        c2: float = 1.0,
        c3: float = 1.0,
        rb_freq: int | None = None,
        m1: float = 10.0,
        m2: float = 2.0,
        m3: float = 0.1,
        optimizer: dict[str, Any] = DEFAULT_OPTIMIZER,
        parameterize: bool = False,
        hat_init: bool = False,
    ) -> None:
        """Mean-field Occupation Measure Optimization algorithm.

        Attributes
        ----------
        L
            Initialization value, used only when `parameterize=False` and
            otherwise ignored.
        z
            Initialization value, used only when `parameterize=False` and
            otherwise ignored.
        y
            Initialization value, used only when `parameterize=False` and
            otherwise ignored.
        u
            Initialization value, used only when `parameterize=True` and
            otherwise ignored.
        v
            Initialization value, used only when `parameterize=True` and
            otherwise ignored.
        w
            Initialization value, used only when `parameterize=True` and
            otherwise ignored.
        loss
            Determines the type of norm used in the objective function.
        c1
            Objective function coefficient.
        c2
            Objective function coefficient.
        c3
            Objective function coefficient.
        rb_freq
            Determines how often residual balancing is applied. If
            None, residual balancing will not be applied.
        m1
            Residual balancing parameter.
        m2
            Residual balancing parameter.
        m3
            Residual balancing parameter.
        optimizer
            Name and configuration of a Pytorch optimizer.
        parameterize
            Optionally solve the alternate "parameterized"
            formulation.
        hat_init
            A boolean determining whether to use hat initialization.
        """
        if loss not in ["l1", "l2", "l1_l2"]:
            raise ValueError("the valid loss arguments are 'l1', 'l2', and 'l1_l2'")
        if rb_freq is not None and rb_freq <= 0:
            raise ValueError("if not None, `rb_freq` must be a positive integer")
        self._L_or_u = _verify(L, u, parameterize=parameterize)
        self._z_or_v = _verify(z, v, parameterize=parameterize)
        self._y_or_w = _verify(y, w, parameterize=parameterize)
        self.loss = loss
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.rb_freq = rb_freq
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.optimizer = optimizer
        self.parameterize = parameterize
        self.hat_init = hat_init

    def _init_L(self, env_instance: Environment, pi: torch.Tensor) -> torch.Tensor:
        if self._L_or_u is None:
            return pi / tuple_prod(env_instance.S)
        else:
            return self._L_or_u

    def _init_u(self, env_instance: Environment, pi: torch.Tensor) -> torch.Tensor:
        # following pi(a|s)=L(s,a)/mu(s), this mean-field yields policy pi_np
        if self._L_or_u is None:
            L = pi / tuple_prod(env_instance.S)
            if (L == 0).any():
                raise ValueError("zero value encountered; unable to take log")
            return torch.log(L)
        else:
            return self._L_or_u

    def _init_z(self, env_instance: Environment, L: torch.Tensor) -> torch.Tensor:
        if self._z_or_v is None:
            T = env_instance.T
            n_s = tuple_prod(env_instance.S)
            n_a = tuple_prod(env_instance.A)
            if self.hat_init:
                z_hat, _ = hat_initialization(env_instance, L, self.parameterize)
                return z_hat  # type: ignore[return-value]
            return torch.zeros((T + 1) * n_s * n_a)
        else:
            return self._z_or_v

    def _init_v(self, env_instance: Environment, L: torch.Tensor) -> torch.Tensor:
        if self._z_or_v is None:
            T = env_instance.T
            n_s = tuple_prod(env_instance.S)
            n_a = tuple_prod(env_instance.A)
            if self.hat_init:
                v_hat, _ = hat_initialization(env_instance, L, self.parameterize)
                if v_hat is not None:
                    return v_hat
            return torch.zeros((T + 1) * n_s * n_a + 1)
        else:
            return self._z_or_v

    def _init_y(self, env_instance: Environment, L: torch.Tensor) -> torch.Tensor:
        if self._y_or_w is None:
            T = env_instance.T
            n_s = tuple_prod(env_instance.S)
            if self.hat_init:
                _, y_hat = hat_initialization(env_instance, L, self.parameterize)
                return y_hat
            return torch.zeros((T + 1) * n_s)
        else:
            return self._y_or_w

    def _init_w(self, env_instance: Environment, L: torch.Tensor) -> torch.Tensor:
        if self._y_or_w is None:
            T = env_instance.T
            n_s = tuple_prod(env_instance.S)
            if self.hat_init:
                _, w_hat = hat_initialization(env_instance, L, self.parameterize)
                return w_hat
            return torch.zeros((T + 1) * n_s)
        else:
            return self._y_or_w

    def __str__(self) -> str:
        """Represent algorithm instance and associated parameters with a string."""
        return (
            f"MFOMO(loss={self.loss}, c1={self.c1}, c2={self.c2}, c3={self.c3}, "
            f"rb_freq={self.rb_freq}, m1={self.m1}, m2={self.m2}, m3={self.m3}, "
            f"parameterize={self.parameterize})"
        )

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
        """Mean-Field Occupation Measure Optimization algorithm.

        Args
        ----
        env_instance
            An instance of a specific environment.
        pi
            A user-provided array of size (T+1,) + S + A representing the initial
            policy. If 'uniform', the initial policy will be uniformly distributed.
        max_iter
            Maximum number of iterations to run.
        atol
            Absolute tolerance criteria for early stopping.
        rtol
            Relative tolerance criteria for early stopping.
        verbose
            Print convergence information during iteration.
        """
        T = env_instance.T
        S = env_instance.S
        A = env_instance.A

        pi = _ensure_free_tensor(pi, env_instance)

        # Initialization
        L = self._init_L(env_instance, pi)
        L_u_tensor = (
            self._init_u(env_instance, pi)
            if self.parameterize
            else self._init_L(env_instance, pi)
        )
        z_v_tensor = (
            self._init_v(env_instance, L)
            if self.parameterize
            else self._init_z(env_instance, L)
        )
        y_w_tensor = (
            self._init_w(env_instance, L)
            if self.parameterize
            else self._init_y(env_instance, L)
        )

        c1, c2 = self.c1, self.c2

        # Instantiate optimizer
        L_u_tensor.requires_grad = True
        z_v_tensor.requires_grad = True
        y_w_tensor.requires_grad = True
        optimizer_to_call = getattr(torch.optim, self.optimizer["name"])
        optimizer_instance = optimizer_to_call(
            params=[L_u_tensor, z_v_tensor, y_w_tensor],
            **self.optimizer.get("config", {}),
        )

        soft_max = torch.nn.Softmax(dim=-1)

        if not self.parameterize:
            pi = mf_omo_policy(env_instance, L_u_tensor.clone().detach())
        else:
            pi = mf_omo_policy(
                env_instance,
                soft_max(L_u_tensor.clone().detach().flatten(start_dim=1)).reshape(
                    (T + 1,) + S + A
                ),
            )

        solutions = [pi]
        argmin = 0
        scores = [exploitability_score(env_instance, pi)]
        runtimes = [0.0]

        if verbose:
            _print_fancy_header(
                alg_instance=self,
                env_instance=env_instance,
                max_iter=max_iter,
                atol=atol,
                rtol=rtol,
            )
            _print_fancy_table_row(
                n=0,
                score_n=scores[0],
                score_0=scores[0],
                argmin=argmin,
                runtime_n=runtimes[0],
            )

        if _trigger_early_stopping(scores[0], scores[0], atol, rtol):
            if verbose:
                _print_solve_complete(seconds_elapsed=runtimes[0])
            return solutions, scores, runtimes

        t = time.time()
        for n in range(1, max_iter + 1):
            # Residual Balancing
            if self.rb_freq and (n + 1) % self.rb_freq == 0:
                c1, c2 = mf_omo_residual_balancing(
                    env_instance,
                    L_u_tensor,
                    z_v_tensor,
                    y_w_tensor,
                    self.loss,
                    c1,
                    c2,
                    self.m1,
                    self.m2,
                    self.m3,
                    self.parameterize,
                )

            # Update the parameters
            obj = mf_omo_obj(
                env_instance,
                L_u_tensor,
                z_v_tensor,
                y_w_tensor,
                self.loss,
                c1,
                c2,
                self.c3,
                self.parameterize,
            )

            # Compute the gradients, update the params, and aply the constraints
            optimizer_instance.zero_grad()
            obj.backward(retain_graph=False)  # type: ignore[no-untyped-call]
            optimizer_instance.step()

            # Constraint enforcement
            if not self.parameterize:
                (
                    L_u_tensor.data,
                    z_v_tensor.data,
                    y_w_tensor.data,
                ) = mf_omo_constraints(env_instance, L_u_tensor, z_v_tensor, y_w_tensor)

            # Compute and store solution policy
            if not self.parameterize:
                pi = mf_omo_policy(env_instance, L_u_tensor.clone().detach())
            else:
                pi = mf_omo_policy(
                    env_instance,
                    soft_max(L_u_tensor.clone().detach().flatten(start_dim=1)).reshape(
                        (T + 1,) + S + A
                    ),
                )

            solutions.append(pi.clone().detach())
            scores.append(exploitability_score(env_instance, pi))
            if scores[n] < scores[argmin]:
                argmin = n
            runtimes.append(time.time() - t)

            if verbose:
                _print_fancy_table_row(
                    n=n,
                    score_n=scores[n],
                    score_0=scores[0],
                    argmin=argmin,
                    runtime_n=runtimes[n],
                )

            if _trigger_early_stopping(scores[0], scores[n], atol, rtol):
                if verbose:
                    _print_solve_complete(seconds_elapsed=runtimes[n])
                return solutions, scores, runtimes

        if verbose:
            _print_solve_complete(seconds_elapsed=time.time() - t)

        return solutions, scores, runtimes

    @classmethod
    def _tuner_instance(cls, trial: optuna.Trial) -> MFOMO:
        rb_freq_bool = trial.suggest_categorical("rb_freq_bool", [False, True])
        rb_freq_num = trial.suggest_int("rb_freq_num", 1, 201, step=10)
        rb_freq = None if rb_freq_bool else rb_freq_num
        optimizer = {
            "name": trial.suggest_categorical("name", ["SGD", "Adam"]),
            "config": {
                "lr": trial.suggest_float("lr", 1e-3, 1e3, log=True),
            },
        }

        return MFOMO(
            loss=trial.suggest_categorical(  # type: ignore[arg-type]
                "loss", ["l1", "l2", "l1_l2"]
            ),
            c1=trial.suggest_float("c1", 1e-2, 1e2, log=True),
            c2=trial.suggest_float("c2", 1e-2, 1e2, log=True),
            rb_freq=rb_freq,
            m1=trial.suggest_float("m1", 10, 100, step=90),
            m2=trial.suggest_float("m2", 2, 4, step=2),
            m3=trial.suggest_float("m3", 0.01, 1.0, step=0.09),
            optimizer=optimizer,
            parameterize=trial.suggest_categorical("parameterize", [False, True]),
            hat_init=trial.suggest_categorical("hat_init", [False, True]),
        )

    def tune(
        self,
        env_suite: list[Environment],
        *,
        max_iter: int = 100,
        atol: float = 1e-3,
        rtol: float = 1e-3,
        metric: Literal["shifted_geo_mean", "failure_rate"] = "shifted_geo_mean",
        n_trials: int | None = 10,
        timeout: float = 30.0,
    ) -> MFOMO:
        """Tune the algorithm over a given environment suite.

        Args
        ----
        env_suite
            A list of environment instances.
        max_iter
            The number of iterations to run the algorithm on each environment
            instance.
        atol
            Absolute tolerance criteria for early stopping.
        rtol
            Relative tolerance criteria for early stopping.
        metric
            Determines which metric to be used for scoring a trial. Either
            ``shifted_geo_mean`` or ``failure_rate``.
        n_trials
            The number of trials. If this argument is not given, as many
            trials are run as possible.
        timeout
            Stop tuning after the given number of second(s) on each
            environment instance. If this argument is not given, as many trials are
            run as possible.
        """
        params = self._optimize_optuna_study(
            env_suite=env_suite,
            max_iter=max_iter,
            atol=atol,
            rtol=rtol,
            metric=metric,
            n_trials=n_trials,
            timeout=timeout,
        )
        if params:
            self.loss = params["loss"]
            self.c1 = params["c1"]
            self.c2 = params["c2"]
            self.rb_freq = None if params["rb_freq_bool"] else params["rb_freq_num"]
            self.m1 = params["m1"]
            self.m2 = params["m2"]
            self.m3 = params["m3"]
            self.optimizer = {
                "name": params["name"],
                "config": {
                    "lr": params["lr"],
                },
            }
            self.parameterize = params["parameterize"]
            self.hat_init = params["hat_init"]
        return self
