from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Literal

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import optuna
import torch

from mfglib.alg.abc import (
    DEFAULT_ATOL,
    DEFAULT_MAX_ITER,
    DEFAULT_RTOL,
    Algorithm,
    Logger,
)
from mfglib.alg.mf_omo_constraints import mf_omo_constraints
from mfglib.alg.mf_omo_obj import mf_omo_obj
from mfglib.alg.mf_omo_residual_balancing import mf_omo_residual_balancing
from mfglib.alg.utils import (
    _ensure_free_tensor,
    _trigger_early_stopping,
    extract_policy_from_mean_field,
    hat_initialization,
)
from mfglib.env import Environment
from mfglib.scoring import exploitability_score

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
    r"""
    **MF-OMO**, or Mean-Field Occupation Measure Inclusion, reformulates the
    MFG as an optimization problem.

    In its most basic form **MF-OMO** solves

    .. math::

        \text{minimize}_{L, y, z} \quad& \left \lVert A_L L - b \right \rVert_2^2 + \left \lVert A_L^\top y + z - c_L \right \rVert_2^2 + z^\top L \\
        \text{subject to} \quad& \mathbf{1}^\top L_t = 0 \quad \forall t \in \mathcal{T} \\
        & \mathbf{1}^\top z \leq SA(T^2 + T + 2) r_{\max} \\
        & \left \lVert y \right \rVert_2 \leq S ( T + 1 ) ( T + 2 ) r_{\max} / 2 \\
        & L \in \mathbb{R}_+^{\mathcal{T} \mathcal{S} \mathcal{A}} \\
        & y \in \mathbb{R}^{\mathcal{T} \mathcal{S}} \\
        & z \in \mathbb{R}^{\mathcal{T} \mathcal{S} \mathcal{A}}

    This constrained optimization problem can be solved with a variety of methods, such
    as Projected Gradient Descent.

    **Parameterized Formulation.** Replacing the constrained optimization problem with a smooth unconstrained
    problem enables us to use a broader range of optimization solvers. As explained in Appendix A.3[#], we can
    reparameterize the variables in **MF-OMO** to completely eliminate the constraints. The new problem is called
    the “parameterized” formulation. Pass ``parameterize=True`` to use this second formulation.

    **Hat Initialization.** Given an initial mean-field :math:`L`, the "hat initialization" sets the initial
    :math:`y,z` to :math:`\hat{y}(L), \hat{z}(L)` as explained in Proposition 6[#]. Pass ``hat_init=True`` to
    enable this option.

    **Redesigned Objective.** One can also assign different coefficients to the
    three terms in the objective function, and come up with a "redesigned objective"

    .. math::

        c_1 \left \lVert A_L L - b \right \rVert_2^2 + c_2 \left \lVert A_L^\top y + z - c_L \right \rVert_2^2 + c_3 z^\top L

    .. note::

        Without loss of generality we can always let :math:`c_1 = 1`.

    You can also apply different norm to the objective terms. The pre-configured options are
     * ``"l1"`` with objective :math:`c_1 \left \lVert A_L L - b \right \rVert_1 + c_2 \left \lVert A_L^\top y + z - c_L \right \rVert_1 + c_3 z^\top L`
     * ``"l2"`` with objective :math:`c_1 \left \lVert A_L L - b \right \rVert_2^2 + c_2 \left \lVert A_L^\top y + z - c_L \right \rVert_2^2 + c_3 ( z^\top L)^2`
     * ``"l1_l2"`` with objective :math:`c_1 \left \lVert A_L L - b \right \rVert_2^2 + c_2 \left \lVert A_L^\top y + z - c_L \right \rVert_2^2 + c_3 z^\top L`

    **Adaptive Residual Balancing:** We can adaptively change the coefficients (:math:`c_1`, :math:`c_2`, and :math:`c_3`)
    of the redesigned objective based on the value of their corresponding objective term. This process is controlled
    by the three parameters, ``m1``, ``m2``, and ``m3``.

    Let's denote by :math:`O_1` the value of the first objective term (depending on the norm used, it could be either
    :math:`\left \lVert A_L L-b \right \rVert_1` or :math:`\lVert A_L L-b \rVert_2^2`), and let :math:`O_2` and
    :math:`O_3` be the values of the second and third objective terms, respectively. When adaptive residual balancing
    is applied, we modify the coefficients in the following way:

    1. If :math:`O_1/ \max \{ O_2, O_3 \} > m_1`, then multiply ``c1`` by ``m2``.
    2. If :math:`O_1/ \min \{ O_2, O_3 \} < m_3`, then divide ``c1`` by ``m2``.
    3. If :math:`O_2/ \max\{ O_1, O_3 \} > m_1`, then multiply ``c2`` by ``m2``.
    4. If :math:`O_2/ \max\{ O_1, O_3 \} > m_3`, then divide ``c2`` by ``m2``.

    ``rb_freq`` determines how frequently the residual rebalancing is applied.

    **Initialization:** We can set the initial policy for any algorithm using the input argument ``pi`` through the `
    `solve()`` method.  **MF-OMO** uses the initial policy to compute the initial values of the variables :math:`L`,
    :math:`z`, and :math:`y`. However, if you want to initialize these variables directly, you can do so by passing
    the ``L``, ``y``, and ``z`` parameters to the constructor. If you're using the parameterized version, you should
    instead pass ``u``, ``v``, and ``w``.

    Parameters
    ----------
    L
        Initialization value, used only when ``parameterize=False`` and
        otherwise ignored.
    z
        Initialization value, used only when ``parameterize=False`` and
        otherwise ignored.
    y
        Initialization value, used only when ``parameterize=False`` and
        otherwise ignored.
    u
        Initialization value, used only when ``parameterize=True`` and
        otherwise ignored.
    v
        Initialization value, used only when ``parameterize=True`` and
        otherwise ignored.
    w
        Initialization value, used only when ``parameterize=True`` and
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
        Name and configuration of a ``pytorch`` optimizer.
    parameterize
        Optionally solve the alternate "parameterized"
        formulation.
    hat_init
        A boolean determining whether to use hat initialization.

    .. seealso::

        Refer to :cite:t:`guo2022` for additional details.
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
        hat_init: bool = True,
    ) -> None:
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

    def save(self, path: Path | str) -> None:
        path = Path(path) / self.__class__.__name__
        path.mkdir(exist_ok=False)
        with open(path / "kwargs.json", "w") as f:
            json_kwargs = {
                "loss": self.loss,
                "c1": self.c1,
                "c2": self.c2,
                "c3": self.c3,
                "rb_freq": self.rb_freq,
                "m1": self.m1,
                "m2": self.m2,
                "m3": self.m3,
                "optimizer": self.optimizer,
                "parameterize": self.parameterize,
                "hat_init": self.hat_init,
            }
            json.dump(json_kwargs, f, indent=4)
        if self.parameterize:
            torch.save(self._L_or_u, path / "u.pt")
            torch.save(self._z_or_v, path / "v.pt")
            torch.save(self._y_or_w, path / "w.pt")
        else:
            torch.save(self._L_or_u, path / "L.pt")
            torch.save(self._z_or_v, path / "z.pt")
            torch.save(self._y_or_w, path / "y.pt")

    @classmethod
    def load(cls, path: Path | str) -> MFOMO:
        path = Path(path) / cls.__name__
        with open(path / "kwargs.json", "r") as f:
            kwargs = json.load(f)
        tensors = {}
        if kwargs["parameterize"]:
            tensors["u"] = torch.load(path / "u.pt")
            tensors["v"] = torch.load(path / "v.pt")
            tensors["w"] = torch.load(path / "w.pt")
        else:
            tensors["L"] = torch.load(path / "L.pt")
            tensors["z"] = torch.load(path / "z.pt")
            tensors["y"] = torch.load(path / "y.pt")
        return MFOMO(**kwargs, **tensors)

    def _init_L(self, env_instance: Environment, pi: torch.Tensor) -> torch.Tensor:
        if self._L_or_u is None:
            return pi / env_instance.n_states
        else:
            return self._L_or_u

    def _init_u(self, env_instance: Environment, pi: torch.Tensor) -> torch.Tensor:
        # following pi(a|s)=L(s,a)/mu(s), this mean-field yields policy pi_np
        if self._L_or_u is None:
            L = pi / env_instance.n_states
            if (L == 0).any():
                raise ValueError("zero value encountered; unable to take log")
            return torch.log(L)
        else:
            return self._L_or_u

    def _init_z(self, env_instance: Environment, L: torch.Tensor) -> torch.Tensor:
        if self._z_or_v is None:
            T = env_instance.T
            n_s = env_instance.n_states
            n_a = env_instance.n_actions
            if self.hat_init:
                z_hat, _ = hat_initialization(env_instance, L, self.parameterize)
                return z_hat  # type: ignore[return-value]
            return torch.zeros((T + 1) * n_s * n_a)
        else:
            return self._z_or_v

    def _init_v(self, env_instance: Environment, L: torch.Tensor) -> torch.Tensor:
        if self._z_or_v is None:
            T = env_instance.T
            n_s = env_instance.n_states
            n_a = env_instance.n_actions
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
            n_s = env_instance.n_states
            if self.hat_init:
                _, y_hat = hat_initialization(env_instance, L, self.parameterize)
                return y_hat
            return torch.zeros((T + 1) * n_s)
        else:
            return self._y_or_w

    def _init_w(self, env_instance: Environment, L: torch.Tensor) -> torch.Tensor:
        if self._y_or_w is None:
            T = env_instance.T
            n_s = env_instance.n_states
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
            f"parameterize={self.parameterize}, hat_init={self.hat_init})"
        )

    def solve(
        self,
        env: Environment,
        *,
        pi_0: Literal["uniform"] | torch.Tensor = "uniform",
        max_iter: int = DEFAULT_MAX_ITER,
        atol: float | None = DEFAULT_ATOL,
        rtol: float | None = DEFAULT_RTOL,
        verbose: bool = False,
        print_every: int = 50,
    ) -> tuple[list[torch.Tensor], list[float], list[float]]:
        """Mean-Field Occupation Measure Optimization algorithm.

        Args
        ----
        env
            An instance of a specific environment.
        pi_0
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
        print_every
            Control how many iterations between printouts.
        """
        T = env.T
        S = env.S
        A = env.A

        pi = _ensure_free_tensor(pi_0, env)

        # Initialization
        L = self._init_L(env, pi)
        L_u_tensor = (
            self._init_u(env, pi) if self.parameterize else self._init_L(env, pi)
        )
        z_v_tensor = self._init_v(env, L) if self.parameterize else self._init_z(env, L)
        y_w_tensor = self._init_w(env, L) if self.parameterize else self._init_y(env, L)

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
            pi = extract_policy_from_mean_field(env, L_u_tensor.clone().detach())
        else:
            pi = extract_policy_from_mean_field(
                env,
                soft_max(L_u_tensor.clone().detach().flatten(start_dim=1)).reshape(
                    (T + 1,) + S + A
                ),
            )

        pis = [pi]
        argmin = 0
        expls = [exploitability_score(env, pi)]
        rts = [0.0]
        # TODO: remove the following line
        obj_vals = torch.empty(max_iter + 1)

        logger = Logger(verbose)
        logger.display_info(
            env=env,
            cls=f"{self.__class__.__name__}",
            parameters={
                "loss": self.loss,
                "c1": self.c1,
                "c2": self.c2,
                "c3": self.c3,
                "rb_freq": self.rb_freq,
                "m1": self.m1,
                "m2": self.m2,
                "m3": self.m3,
                "parameterize": self.parameterize,
                "hat_init": self.hat_init,
                "optimizer": self.optimizer,
            },
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
            # Residual Balancing
            if self.rb_freq and (i + 1) % self.rb_freq == 0:
                c1, c2 = mf_omo_residual_balancing(
                    env,
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
                env,
                L_u_tensor,
                z_v_tensor,
                y_w_tensor,
                self.loss,
                c1,
                c2,
                self.c3,
                self.parameterize,
            )
            # TODO: remove the following line
            obj_vals[i - 1] = obj.item()

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
                ) = mf_omo_constraints(env, L_u_tensor, z_v_tensor, y_w_tensor)

            # Compute and store solution policy
            if not self.parameterize:
                pi = extract_policy_from_mean_field(env, L_u_tensor.clone().detach())
            else:
                pi = extract_policy_from_mean_field(
                    env,
                    soft_max(L_u_tensor.clone().detach().flatten(start_dim=1)).reshape(
                        (T + 1,) + S + A
                    ),
                )

            pis.append(pi.clone().detach())
            expls.append(exploitability_score(env, pi))
            rts.append(time.time() - t_0)
            if expls[i] < expls[argmin]:
                argmin = i
            if i % print_every == 0:
                logger.insert_row(
                    i=i,
                    expl=expls[i],
                    ratio=expls[i] / expls[0],
                    argmin=argmin,
                    elapsed=rts[i],
                )
            if _trigger_early_stopping(expls[0], expls[i], atol, rtol):
                if i % print_every != 0:
                    logger.insert_row(
                        i=i,
                        expl=expls[i],
                        ratio=expls[i] / expls[0],
                        argmin=argmin,
                        elapsed=rts[i],
                    )
                logger.flush_stopped()
                return pis, expls, rts

        logger.insert_row(
            i=max_iter,
            expl=expls[max_iter],
            ratio=expls[max_iter] / expls[0],
            argmin=argmin,
            elapsed=rts[max_iter],
        )
        logger.flush_exhausted()

        # TODO: uncomment the following line
        # return pis, expls, rts

        obj = mf_omo_obj(
            env,
            L_u_tensor,
            z_v_tensor,
            y_w_tensor,
            self.loss,
            c1,
            c2,
            self.c3,
            self.parameterize,
        )
        obj_vals[max_iter] = obj.item()

        return pis, expls, obj_vals

    def _init_tuner_instance(self: Self, trial: optuna.Trial) -> Self:
        rb_freq_bool = trial.suggest_categorical("rb_freq_bool", [False, True])
        rb_freq_num = trial.suggest_int("rb_freq_num", 1, 201, step=10)
        rb_freq = None if rb_freq_bool else rb_freq_num
        optimizer = {
            "name": trial.suggest_categorical("name", ["SGD", "Adam"]),
            "config": {
                "lr": trial.suggest_float("lr", 1e-3, 1e3, log=True),
            },
        }
        return type(self)(
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

    def from_study(self: Self, study: optuna.Study) -> Self:
        expected_params = {
            "loss",
            "c1",
            "c2",
            "rb_freq_bool",
            "rb_freq_num",
            "m1",
            "m2",
            "m3",
            "parameterize",
            "name",
            "lr",
            "hat_init",
        }
        err_msg = f"{study.best_params.keys()=} != {expected_params}."
        assert study.best_params.keys() == expected_params, err_msg

        best_params = study.best_params
        rb_freq_bool = best_params.pop("rb_freq_bool")
        rb_freq_num = best_params.pop("rb_freq_num")
        rb_freq = None if rb_freq_bool else rb_freq_num

        optimizer = {
            "name": best_params.pop("name"),
            "config": {"lr": best_params.pop("lr")},
        }

        return type(self)(rb_freq=rb_freq, optimizer=optimizer, **best_params)
