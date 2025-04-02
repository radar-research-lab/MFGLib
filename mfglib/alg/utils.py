from __future__ import annotations

import warnings
from functools import reduce
from typing import Literal, TypeVar

import torch
from rich import box
from rich import print as rich_print
from rich.console import Group
from rich.panel import Panel
from rich.pretty import pretty_repr
from rich.table import Table
from rich.text import Text
from typing_extensions import Self

from mfglib import __version__
from mfglib.alg.q_fn import QFn
from mfglib.env import Environment

T = TypeVar("T", int, float)


def tuple_prod(tup: tuple[T, ...]) -> T:
    """Compute product of the elements in a tuple."""
    return reduce(lambda x, y: x * y, tup)


def _new_table() -> Table:
    return Table(
        "Iter (n)",
        "Expl_n",
        "Ratio_n",
        "Argmin_n",
        "Elapsed_n",
    )


class Printer:

    WIDTH = 61

    def __init__(self, verbose: int, expl_0: float) -> None:
        self.verbose = verbose
        self.argmin = 0
        self.expl_argmin = expl_0
        self.expl_0 = expl_0

        self.table = _new_table()

    @classmethod
    def setup(
        cls,
        verbose: int,
        env_instance: Environment,
        solver: str,
        parameters: dict[str, float | str | None | int],
        atol: float | None,
        rtol: float | None,
        max_iter: int,
        expl_0: float,
    ) -> Self:
        printer = cls(verbose, expl_0)
        printer.print_info_panels(
            env_instance, solver, parameters, atol, rtol, max_iter
        )
        printer.notify_of_solution(n=0, expl_n=expl_0, runtime_n=0.0)
        return printer

    def print_info_panels(
        self,
        env_instance: Environment,
        solver: str,
        parameters: dict[str, float | str | None | int],
        atol: float | None,
        rtol: float | None,
        max_iter: int,
    ) -> None:
        if self.verbose > 0:
            top_group = Group(
                Text(
                    f"MFGLib v{__version__} : A Library for Mean-Field Games",
                    justify="center",
                    style="bold",
                ),
                Text("RADAR Research Lab, UC Berkeley", justify="center"),
            )
            top_panel = Panel(top_group, box=box.HEAVY, width=self.WIDTH, padding=1)

            env_group = Group(
                f"S = {env_instance.S}",
                f"A = {env_instance.A}",
                f"T = {env_instance.T}",
                f"r_max = {env_instance.r_max}",
            )
            env_panel = Panel(
                env_group,
                title=Text("Environment Summary", style="bold"),
                width=self.WIDTH,
                title_align="left",
                box=box.SQUARE,
            )

            alg_group = Group(
                f"class = {solver}",
                f"parameters = {pretty_repr(parameters)}",
                f"atol = {atol}",
                f"rtol = {rtol}",
                f"max_iter = {max_iter}",
            )
            alg_panel = Panel(
                alg_group,
                title=Text("Algorithm Summary", style="bold"),
                width=self.WIDTH,
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
                width=self.WIDTH,
                title_align="left",
                box=box.SQUARE,
            )
            rich_print(top_panel, env_panel, alg_panel, doc_panel)

    def notify_of_solution(self, n: int, expl_n: float, runtime_n: float) -> None:
        if expl_n < self.expl_argmin:
            self.argmin = n
            self.expl_argmin = expl_n

        if self.verbose > 0 and n % self.verbose == 0:
            self.table.add_row(
                f"{n}",
                f"{expl_n:.4e}",
                f"{expl_n / self.expl_0:.4e}",
                f"{self.argmin}",
                f"{runtime_n:.2e}",
            )
            if len(self.table.rows) == 50:
                rich_print(self.table)
                self.table = _new_table()

    def alert_early_stopping(self) -> None:
        if self.verbose > 0:
            rich_print(self.table)
            rich_print("Absolute or relative stopping criteria met.")

    def alert_iterations_exhausted(self) -> None:
        if self.verbose > 0:
            rich_print(self.table)
            rich_print("Number of iterations exhausted.")


def _ensure_free_tensor(
    pi: Literal["uniform"] | torch.Tensor, env_instance: Environment
) -> torch.Tensor:
    """Construct uniform tensor if necessary.

    By free, we mean cloned and detached.
    """
    if pi == "uniform":
        T, S, A = env_instance.T, env_instance.S, env_instance.A
        return torch.ones((T + 1,) + S + A) / torch.ones(A).sum()
    elif isinstance(pi, torch.Tensor):
        return pi.clone().detach()
    else:
        raise TypeError(f"unexpected type {type(pi)} for pi")


def _trigger_early_stopping(
    score_0: float, score_n: float, atol: float | None, rtol: float | None
) -> bool:
    if atol or rtol:
        atolv = 0.0 if atol is None else atol
        rtolv = 0.0 if rtol is None else rtol

        return score_n <= atolv + rtolv * score_0
    else:
        return False


def project_onto_simplex(
    x: torch.Tensor,
    r: float = 1.0,
) -> torch.Tensor:
    """Project x onto a simplex with upper bound r using sorting.

    Notes
    -----
    See [1]_ for details.

    .. [1] Efficient: Duchi et al (2008). "Efficient Projections onto the l1-Ball for
    Learning in High Dimensions." Fig. 1 and Sect. 4.
    https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
    """
    x_d = torch.atleast_1d(x).double()  # type: ignore[no-untyped-call]
    if r < 0.0:
        raise ValueError("r must be a non-negative scalar.")
    elif r == 0:
        return torch.zeros_like(x)
    x_decr, _ = torch.sort(x_d, descending=True)
    x_cumsum = torch.cumsum(x_decr, dim=-1)
    denom = torch.arange(1, len(x_decr) + 1)
    theta = (x_cumsum - r) / denom
    x_diff = x_decr - theta
    if (x_diff > 0).any():
        idx = torch.max(torch.argwhere(x_diff > 0).ravel())
        c_star = theta[idx]
        x_p = torch.maximum(x_d - c_star, torch.tensor(0)).to(x.dtype)
        return x_p / x_p.sum() * r
    else:
        warnings.warn(
            "failed to project onto the simplex; "
            "reset to r * e / len(x), where e is the 1-vector of size len(x)"
        )
        return torch.ones(len(x)) / len(x) * r


def extract_policy_from_mean_field(
    env_instance: Environment, L: torch.Tensor
) -> torch.Tensor:
    """Compute the policy given mean-field.

    Args:
    ----
    env_instance: An instance of a specific environment.
    L: A numpy array/tensor of size (T+1,)+S+A representing the mean-field L.
    """
    # Environment parameters
    S = env_instance.S  # state space dimensions
    A = env_instance.A  # action space dimensions

    # Auxiliary variables
    l_s = len(S)
    l_a = len(A)
    n_a = tuple_prod(A)
    ones_ts = (1,) * (l_s + 1)
    ats_to_tsa = tuple(range(l_a, l_a + 1 + l_s)) + tuple(range(l_a))

    # Corresponding policy

    L_sum_rptd = (
        L.flatten(start_dim=1 + l_s).sum(-1).repeat(A + ones_ts).permute(ats_to_tsa)
    )
    pi = L.div(L_sum_rptd).nan_to_num(
        nan=1 / n_a, posinf=1 / n_a, neginf=1 / n_a
    )  # using uniform distribution when L_t_sum_rptd is zero

    return pi


def hat_initialization(
    env_instance: Environment,
    L: torch.Tensor,
    parameterize: bool,
    z_eps: float = 1e-8,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Initialize hat vectors."""
    # Environment parameters
    T = env_instance.T
    S = env_instance.S
    A = env_instance.A
    r_max = env_instance.r_max

    # Auxiliary parameters
    n_s = tuple_prod(S)
    n_a = tuple_prod(A)
    l_a, l_s = len(A), len(S)
    ones_s = (1,) * l_s
    as_to_sa = tuple(range(l_a, l_a + l_s)) + tuple(range(l_a))
    ssa_to_sas = tuple(range(l_s, l_s + l_s + l_a)) + tuple(range(l_s))
    c_v = n_s * n_a * (T**2 + T + 2) * r_max
    c_w = n_s * (T + 1) * (T + 2) * r_max / 2 / torch.sqrt(torch.tensor(n_s * (T + 1)))

    # Compute V*
    qfn = QFn(env_instance, L)
    q_star = qfn.optimal()
    v, _ = q_star.flatten(start_dim=1 + l_s).max(dim=-1)
    v_fltn = v.clone().detach().flatten()

    # y_hat
    y_hat = torch.cat((v_fltn[n_s:], -v_fltn[:n_s]))

    # z_hat
    z_hat = torch.zeros((T + 1,) + S + A)
    for t in range(T):
        v_t = v[t].repeat(A + ones_s).permute(as_to_sa)
        r_t = env_instance.reward(t, L[t])
        p_t = (
            env_instance.prob(t, L[t]).permute(ssa_to_sas).flatten(start_dim=l_s + l_a)
        )
        z_hat[t] = v_t - r_t - p_t @ v[t + 1].flatten()
    z_hat[T] = v[T].repeat(A + ones_s).permute(as_to_sa) - env_instance.reward(T, L[T])
    z_hat = z_hat.clone().detach().flatten()
    z_hat[z_hat < 0] = 0.0

    if not parameterize:
        return z_hat, y_hat

    # v_hat
    z_tild = z_hat / c_v
    v_hat = None
    if z_tild.sum() < 1:
        z_new = 1 - z_tild.sum()
        z_aug = torch.cat((z_tild, 1 - z_new.unsqueeze(dim=-1)))
        z_aug[z_aug == 0.0] = z_eps
        v_hat = torch.log(z_aug)

    # w_hat
    w_hat = y_hat / c_w
    w_hat = torch.asin(w_hat)

    return v_hat, w_hat
