Quickstart
==========

Installation
------------

MFGLib provides pre-built wheels for Python 3.9+ and can be installed via ``pip`` on
all major platforms.

.. code-block::

   $ pip install mfglib

Additionally, the source code is directly available on `GitHub <https://github.com/radar-research-lab/MFGLib>`_.

Introduction
------------

MFGs
^^^^

Informal
""""""""

Multi-agent systems are ubiquitous in modern engineering applications. However, *large-population* systems
present a challenge. Traditional :math:`N`-players methods eventually break down as :math:`N \to \infty`.

An MFG is a type of mathematical model capable of describing systems with many, many agents. To accomplish this,
we must assume

1. The agents/players are indistinguishable and
2. Any individual agent is negligible relative to the population.

These assumptions allow us to reduce the game to studying the interactions between an arbitrary representative
agent and the population as a whole.

Formal
""""""

In what follows, :math:`\Delta_{X}` denotes the set of probability distributions over a given finite set :math:`X`. Furthermore,
if :math:`X` and :math:`Y` are two given finite sets, then :math:`X^Y` denotes the set of maps from :math:`Y` to :math:`X`.

Let :math:`\mathcal{T} \triangleq \{0, 1, \dots, T\}` denote the set of timesteps, :math:`\mathcal{S} \triangleq \{1, 2, \dots, S\}`
be the state space, and :math:`\mathcal{A} \triangleq \{1, 2, \dots, A\}` be the action space. We describe the representative agent
by a policy :math:`\pi \in \left( \Delta_{\mathcal{A}} \right)^{\mathcal{T} \times \mathcal{S}}`,
which assigns a distribution over actions to any timestep and state. We describe the population by a time-indexed joint state-action
distribution denoted :math:`L \in \left( \Delta_{\mathcal{S} \times \mathcal{A}} \right)^{\mathcal{T}}`. We refer to :math:`L`
as the "mean-field".

Given a fixed mean-field :math:`L`, the representative agent faces a standard Markov Decision Process, ie. she
seeks a policy :math:`\pi` to maximize her value function

.. math::

    V(\pi; L) \triangleq \mathbb{E}\left[ \sum_{t=0}^{T} \gamma^t r_t(s_t, a_t, L_t) \mid s_0 \sim \mu_0 \right]

subject to dynamics :math:`a_t \sim \pi_t(s_t)` and :math:`s_{t + 1} \sim P(s_t, a_t, L_t)`. If every agent
were to follow policy :math:`\pi` it would induce a mean-field we denote by :math:`L^{\pi}`.

To "solve" a MFG we seek a Nash equilibrium (NE). A pair :math:`\left( \pi^*, L^* \right)` is a NE
if it meets two conditions:

1. (optimality) :math:`\pi^* \in \arg \max_{\pi} V(\pi; L^*)`
2. (consistency) :math:`L^* = L^{\pi^*}`

Example
^^^^^^^

In this section we describe a very simple MFG environment known as **Left Right**:cite:p:`pmlr-v130-cui21a`.

We let a large number of agents simultaneously choose between going left :math:`(\ell)` or right :math:`(\varrho)`. Afterwards,
each agent shall be punished proportional to the number of agents that chose the same action, but more-so
for choosing right than left.

Formally, let :math:`\mathcal{T} = \{0, 1\}, \mathcal{S} = \{c, \ell, \varrho \}, \mathcal{A} = \mathcal{S} \setminus \{c\}, \mu_0(c) = 1`
and define the reward function

.. math::

    r_t(s_t, a_t, L_t) = - \mathbf{1}_{\{ \ell \}} (s_t) \mu_t(\ell) - 2 \cdot \mathbf{1}_{\{ \varrho \}} (s_t) \mu_t(\varrho)

Note :math:`\mu_t(s) \triangleq \sum_{a \in \mathcal{A}} L_t(s, a)` denotes the proportion of the population occupying state :math:`s` at time :math:`t`.
The transition kernel allows picking the next state directly, ie. for all :math:`s, s' \in \mathcal{S}` and
:math:`a \in \mathcal{A}`

.. math::

    \Pr(s_{t + 1} = s' \mid s_t = s, a_t = a) = \mathbf{1}_{\{ s' \}}(a)

One can verify that any NE policy :math:`\pi^* = (\pi_0^*, \pi_1^*)` must satisfy

.. math::

    \pi_0^*(c) \equiv \begin{bmatrix} \Pr(a_t = \ell \mid s_t = c) \\ \Pr(a_t = \varrho \mid s_t = c) \end{bmatrix} = \begin{bmatrix} \frac{2}{3} \\ \frac{1}{3} \end{bmatrix}

Exploitability
^^^^^^^^^^^^^^

For most MFGs, we may not be able to get a closed-form of their NE solutions. Therefore, in order to find how close a
given policy is to an NE solution, we use a metric called **exploitability**.

Exploitability characterizes the suboptimality of a policy :math:`\pi` against mean-field :math:`L^{\pi}` as follows:

.. math::

    \text{Expl}(\pi) \triangleq \max_{\pi'} V(\pi'; L^{\pi}) - V(\pi; L^{\pi})

If :math:`\text{Expl}(\pi^*) \leq \epsilon` for some :math:`\epsilon \geq 0` then :math:`( \pi^*, L^{\pi^*} )`
is said to be an :math:`\epsilon`-Nash equilibrium. If :math:`\epsilon = 0` then :math:`( \pi^*, L^{\pi^*} )` is
an exact NE. For a more complete description of exploitability, refer to :cite:t:`2022:guo`.
