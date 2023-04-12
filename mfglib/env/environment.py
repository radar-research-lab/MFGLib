from __future__ import annotations

import math
from typing import Literal, Protocol

import torch


# By defining Protocols here, we allow a user to pass any object that implements
# `__call__`. In particular, this allows a user to pass either a function or a class
# for the reward and transition functions.
class RewardFn(Protocol):
    def __call__(self, env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor:
        ...


class TransitionFn(Protocol):
    def __call__(self, env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor:
        ...


class Environment:
    """General environment class."""

    def __init__(
        self,
        *,
        T: int,
        S: tuple[int, ...],
        A: tuple[int, ...],
        mu0: torch.Tensor,
        r_max: float,
        reward_fn: RewardFn,
        transition_fn: TransitionFn,
    ) -> None:
        """General environment class.

        Attributes
        ----------
        T : int
            Time horizon. Time steps will be in :math:`{0, 1, ..., T}`.
        S : tuple[int, ...]
            State space shape.
        A : tuple[int, ...]
            Action space shape.
        mu0 : torch.Tensor
            Initial state distribution with shape ``S``.
        r_max : float
            Supremum of the absolute value of rewards.
        reward_fn : RewardFn
            A function or a class which implements ``__call__`` to compute
            the rewards.
        transition_fn : TransitionFn
            A function or a class which implements ``__call__`` to compute
            the transition probabilities.

        Returns
        -------
        None
        """
        if (mu0.sum() - 1.0).abs() > 1e-6:
            raise ValueError("invalid distribution provided for mu0")
        self.T = T
        self.S = S
        self.A = A
        self.mu0 = mu0
        self.r_max = r_max
        self.reward_fn = reward_fn
        self.transition_fn = transition_fn

    def reward(self, t: int, L_t: torch.Tensor) -> torch.Tensor:
        return self.reward_fn(self, t, L_t)

    def prob(self, t: int, L_t: torch.Tensor) -> torch.Tensor:
        return self.transition_fn(self, t, L_t)

    @classmethod
    def beach_bar(
        cls,
        T: int = 2,
        n: int = 4,
        bar_loc: int = 2,
        log_eps: float = 1e-20,
        p_still: float = 0.5,
        mu0: Literal["uniform"] | torch.Tensor = "uniform",
    ) -> Environment:
        """Beach Bar environment.

        The beach bar process is a Markov Decision Process with :math:`|X|`
        states disposed on a one dimensional torus (:math:`X = {0,..., |X|-1}`), which
        represents a beach. A bar is located in one of the states. As the
        weather is very hot, players want to be as close as possible to the bar,
        while keeping away from too crowded areas.[#1bb]_

        .. [#1bb] Perrin, Sarah, et al. "Fictitious play for mean field games:
            Continuous time analysis and applications." Advances in Neural
            Information Processing Systems 33 (2020): 13199-13213.
        """
        from mfglib.env.examples.beach_bar import RewardFn, TransitionFn

        if bar_loc < 0 or bar_loc > n - 1:
            raise ValueError("bar_loc must be between zero and n-1 (inclusive)")
        if p_still < 0 or p_still > 1:
            raise ValueError("p_still must be a valid probability")

        return cls(
            T=T,
            S=(n,),
            A=(3,),
            mu0=mu0 if isinstance(mu0, torch.Tensor) else torch.ones(n) / n,
            r_max=n,
            reward_fn=RewardFn(n, bar_loc, log_eps),
            transition_fn=TransitionFn(n, p_still),
        )

    @classmethod
    def building_evacuation(
        cls,
        T: int = 3,
        n_floor: int = 5,
        floor_l: int = 10,
        floor_w: int = 10,
        log_eps: float = 1e-20,
        eta: float = 1.0,
        evac_r: float = 10.0,
        mu0: Literal["uniform"] | torch.Tensor = "uniform",
    ) -> Environment:
        """Building Evacuation environment.

        In this problem, there is a multilevel building and each agent of the
        crowd wants to go downstairs as quickly as possible while favoring
        social distancing. At each floor, two staircases are located at two
        opposite corners, such as the crowd has to cross the whole floor to take
        the next staircase. Each agent can remain in place, move in the 4
        directions (up, down, right, left) as well as go up or down when on a
        staircase location.[#be]_

        .. [#be] Perolat, Julien, et al. "Scaling up mean field games with online mirror
            descent." arXiv preprint arXiv:2103.00623 (2021).
        """  # noqa: D401
        from mfglib.env.examples.building_evacuation import RewardFn, TransitionFn

        S = (n_floor, floor_l, floor_w)

        def uniform() -> torch.Tensor:
            return torch.ones(S) / torch.ones(S).sum()

        return cls(
            T=T,
            S=S,
            A=(6,),
            mu0=mu0 if isinstance(mu0, torch.Tensor) else uniform(),
            r_max=-eta * math.log(log_eps) + evac_r,
            reward_fn=RewardFn(S, eta, log_eps, evac_r),
            transition_fn=TransitionFn(n_floor, floor_l, floor_w),
        )

    @classmethod
    def conservative_treasure_hunting(
        cls,
        T: int = 5,
        n: int = 3,
        r: tuple[float, ...] = (1.0, 1.0, 1.0),
        c: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0),
        mu0: Literal["uniform"] | torch.Tensor = "uniform",
    ) -> Environment:
        """Conservative Treasure Hunting environment.

        MF-OMO: An Optimization Formulation of Mean-Field Games
        Guo, X., Hu, A., & Zhang, J. (2022). arXiv:2206.09608.
        """
        from mfglib.env.examples.conservative_treasure_hunting import (
            RewardFn,
            TransitionFn,
        )

        if n != len(r):
            raise ValueError("n must equal len(r)")
        if T != len(c):
            raise ValueError("T must equal len(C)")

        return cls(
            T=T,
            S=(n,),
            A=(n,),
            mu0=mu0 if isinstance(mu0, torch.Tensor) else torch.ones(n) / n,
            r_max=max(r),
            reward_fn=RewardFn(n, r),
            transition_fn=TransitionFn(n, c),
        )

    @classmethod
    def crowd_motion(
        cls,
        T: int = 3,
        torus_l: int = 20,
        torus_w: int = 20,
        loc_change_freq: int = 2,
        c: float = 10.0,
        log_eps: float = 1e-10,
        p_still: float = 0.5,
        seed: int = 0,
        mu0: Literal["uniform"] | torch.Tensor = "uniform",
    ) -> Environment:
        """Crowd Motion environment.

        An adaptation of Crowd Motion environment, which extends the Beach Bar
        environment in 2 dimensions, introduced in

        Perolat, Julien, et al. "Scaling up mean field games with online mirror
        descent." arXiv preprint arXiv:2103.00623 (2021).
        """
        from mfglib.env.examples.crowd_motion import RewardFn, TransitionFn

        S = (torus_l, torus_w)

        def uniform() -> torch.Tensor:
            return torch.ones(S) / torch.ones(S).sum()

        torch.manual_seed(seed)

        return cls(
            T=T,
            S=S,
            A=(5,),
            mu0=mu0 if isinstance(mu0, torch.Tensor) else uniform(),
            r_max=c - math.log(log_eps),
            reward_fn=RewardFn(T, torus_l, torus_w, loc_change_freq, c, log_eps),
            transition_fn=TransitionFn(torus_l, torus_w, p_still),
        )

    @classmethod
    def equilibrium_price(
        cls,
        T: int = 4,
        s_inv: int = 3,
        Q: int = 2,
        H: int = 2,
        d: float = 1.0,
        e0: float = 1.0,
        sigma: float = 1.0,
        c: tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0),
        mu0: Literal["uniform"] | torch.Tensor = "uniform",
    ) -> Environment:
        """Equilibrium Price environment.

        In this problem, a large number of homogeneous firms producing the same
        product under perfect competition are considered. The price of the
        product is determined endogenously by the supply-demand equilibrium.
        Each firm, meanwhile, maintains a certain inventory level of the raw
        materials for production, and decides about the quantity of raw
        materials to consume for production and the quantity of raw materials to
        replenish the inventory.

        Guo, X., Hu, A., Xu, R., & Zhang, J. (2022).
        A general framework for learning mean-field games.
        Mathematics of Operations Research.
        """
        from mfglib.env.examples.equilibrium_price import RewardFn, TransitionFn

        c0, c1, c2, c3, c4 = c

        r_max = (
            (d / e0) ** (1 / sigma) * Q
            + c0 * Q
            + c1 * Q**2
            + c2 * H
            + (c2 + c3) * Q
            + c4 * s_inv
        )

        n_s = s_inv + 1
        return cls(
            T=T,
            S=(n_s,),
            A=(Q + 1, H + 1),
            mu0=mu0 if isinstance(mu0, torch.Tensor) else torch.ones(n_s) / n_s,
            r_max=r_max,
            reward_fn=RewardFn(s_inv, Q, H, d, e0, sigma, c),
            transition_fn=TransitionFn(s_inv, Q, H),
        )

    @classmethod
    def left_right(
        cls, mu0: tuple[float, float, float] = (1.0, 0.0, 0.0)
    ) -> Environment:
        """Left-Right environment.

        A large number of agents choose simultaneously between going left (L) or
        right (R). Afterwards, each agent shall be punished proportional to the
        number of agents that chose the same action, but more-so for choosing right
        than left.

        Cui, Kai, and Heinz Koeppl. "Approximately solving mean field games via
        entropy-regularized deep reinforcement learning." International Conference
        on Artificial Intelligence and Statistics. PMLR, 2021.
        https://proceedings.mlr.press/v130/cui21a.html
        """
        from mfglib.env.examples.left_right import RewardFn, TransitionFn

        return cls(
            T=1,
            S=(3,),
            A=(2,),
            mu0=torch.tensor(mu0),
            r_max=2.0,
            reward_fn=RewardFn(),
            transition_fn=TransitionFn(),
        )

    @classmethod
    def linear_quadratic(
        cls,
        T: int = 3,
        el: int = 5,
        m: int = 2,
        sigma: float = 3.0,
        delta: float = 0.1,
        k: float = 1.0,
        q: float = 0.01,
        kappa: float = 0.5,
        c_term: float = 1.0,
        mu0: Literal["uniform"] | torch.Tensor = "uniform",
    ) -> Environment:
        """Linear Quadratic environment.

        Perrin, Sarah, et al. "Fictitious play for mean field games: Continuous time
        analysis and applications." Advances in Neural Information Processing
        Systems 33 (2020): 13199-13213.
        """
        from mfglib.env.examples.linear_quadratic import RewardFn, TransitionFn

        n_s = 2 * el + 1
        return cls(
            T=T,
            S=(n_s,),
            A=(2 * m + 1,),
            mu0=mu0 if isinstance(mu0, torch.Tensor) else torch.ones(n_s) / n_s,
            r_max=(0.5 * m**2 + 2 * q * m * el + 2 * kappa * el**2) * delta,
            reward_fn=RewardFn(T, el, m, delta, q, kappa, c_term),
            transition_fn=TransitionFn(el, m, sigma, delta, k),
        )

    @classmethod
    def random_linear(
        cls,
        T: int = 3,
        n: int = 5,
        m: float = 10.0,
        seed: int = 0,
        mu0: Literal["uniform"] | torch.Tensor = "uniform",
    ) -> Environment:
        """General linear environment.

        A custom environment in which the rewards and transition probabilities
        are random affine functions of the mean-field. For transition
        probabilities to be valid, a softmax function is applied on top of the
        corresponding affine function.
        """
        from mfglib.env.examples.random_linear import RewardFn, TransitionFn

        torch.manual_seed(seed)

        return cls(
            T=T,
            S=(n,),
            A=(n,),
            mu0=mu0 if isinstance(mu0, torch.Tensor) else torch.ones(n) / n,
            r_max=2 * m,
            reward_fn=RewardFn(n, m),
            transition_fn=TransitionFn(n, m),
        )

    @classmethod
    def rock_paper_scissors(
        cls, T: int = 1, mu0: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    ) -> Environment:
        """Rock-Paper-Scissors environment.

        This game is inspired by Shapley (1964) and their generalized non-zero-sum
        version of Rock-Paper-Scissors, for which classical fictitious play would not
        converge. Each of the agents can choose between rock, paper and scissors, and
        obtains a reward proportional to double the number of beaten agents minus the
        number of agents beating the agent.

        Cui, Kai, and Heinz Koeppl. "Approximately solving mean field games via
        entropy-regularized deep reinforcement learning." International Conference
        on Artificial Intelligence and Statistics. PMLR, 2021.
        https://proceedings.mlr.press/v130/cui21a.html
        """
        from mfglib.env.examples.rock_paper_scissors import RewardFn, TransitionFn

        return cls(
            T=T,
            S=(4,),
            A=(3,),
            mu0=torch.tensor(mu0),
            r_max=6.0,
            reward_fn=RewardFn(),
            transition_fn=TransitionFn(),
        )

    @classmethod
    def susceptible_infected(
        cls, T: int = 50, mu0: tuple[float, float] = (0.4, 0.6)
    ) -> Environment:
        """SIS environment.

        In this problem, a large number of agents can choose between social
        distancing (D) or going out (U). If a susceptible (S) agent chooses social
        distancing, they may not become infected (I). Otherwise, an agent may become
        infected with a probability proportional to the number of agents being infected.
        If infected, an agent will recover with a fixed chance every time step. Both
        social distancing and being infected have an associated cost.

        Cui, Kai, and Heinz Koeppl. "Approximately solving mean field games via
        entropy-regularized deep reinforcement learning." International Conference
        on Artificial Intelligence and Statistics. PMLR, 2021.
        https://proceedings.mlr.press/v130/cui21a.html
        """
        from mfglib.env.examples.susceptible_infected import RewardFn, TransitionFn

        return cls(
            T=T,
            S=(2,),
            A=(2,),
            mu0=torch.tensor(mu0),
            r_max=1.5,
            reward_fn=RewardFn(),
            transition_fn=TransitionFn(),
        )
