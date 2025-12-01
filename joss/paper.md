---
title: "MFGLib: A Library for Mean-Field Games"
tags:
    - mean-field games
    - large-population dynamics
    - multi-agent systems
authors:
    - name: Xin Guo
      affiliation: "1, 2"
    - name: Anran Hu
      affiliation: 3
    - name: Matteo Santamaria
      affiliation: 1
    - name: Mahan Tajrobehkar
      affiliation: 1
    - name: Junzi Zhang
      affiliation: 4
affiliations:
    - name: University of California, Berkeley
      index: 1
    - name: Amazon.com (Amazon Scholar)
      index: 2
    - name: Columbia University
      index: 3
    - name: Citadel Securities
      index: 4
date: 1 December 2025
bibliography: paper.bib
---

# Summary

Mean-field games (MFGs) provide scalable models for large-population strategic interactions. 
They approximate an $N$-player game by analyzing the limiting regime as $N \to \infty$, 
replacing explicit multi-agent interactions with the interaction between a representative agent 
and the population distribution [@lasry:2007]. MFG models have become widely used in applications 
such as digital advertising, high-frequency trading, dynamic pricing, transportation, and behavioral 
modeling.

Despite the rapid growth of the MFG literature, researchers and practitioners lack a unified, 
open-source software package for defining and solving their own MFG problems. Existing MFG 
implementations are largely one-off, built for experiments within individual papers, and are 
not designed for general use or extensibility.

**MFGLib** is an open-source Python library that fills this gap. It provides:

* A modular and extensible API for defining *arbitrary discrete-time finite-state MFGs*
* A set of *implementations of state-of-the-art algorithms* for computing (approximate) Nash equilibria
* A collection of *pre-built benchmark environments* drawn from the literature
* *Automatic hyperparameter tuning* for all included algorithms
* Clear documentation and examples, enabling both research and industry use

# Statement of need

Over the past decade, the study of numerical methods for MFGs has grown rapidly, with new 
algorithms proposed across control theory, economics, reinforcement learning, and operations 
research. However, the field currently lacks a standardized and user-friendly software tool 
analogous to **OpenSpiel** for general games or **CVXPY** for convex optimization.

Existing tools fall short for one of two reasons:

1. **General game libraries** such as *Nashpy*, *QuantEcon*, and *OpenSpiel* do not provide flexible support for mean-field interaction structures, population distributions, or MFG-specific solvers.

2. **MFG-specific repositories** such as *gmfg-learning* or *entropic-mfg* are designed for reproducing a single publication's experiments. They do not expose reusable abstractions, extensible environment definitions, or a stable API.

As a result, researchers re-implement environments, MFG solvers, and population dynamics from scratch—an error-prone and time-consuming process that hinders reproducibility and comparison of new algorithms.

**MFGLib addresses this need by providing the first general-purpose, customizable, and well-documented MFG library.**
It lowers the barrier to entry for new researchers, enables rapid experimentation, and offers practitioners a way to prototype MFG-based models without requiring background in game theory or optimal control. An early internal version has already been adopted in production systems at Amazon Advertising.

MFGLib therefore fills an important gap in the scientific software ecosystem for large-population games.

# Features and functionality

## 1. Modular library design

MFGLib is organized around two independent core modules:

* **Environments:** define the MFG (state space, action space, rewards, transitions, and population dynamics).
* **Algorithms:** implement solvers that compute approximate Nash equilibria.

Any environment can be paired with any algorithm, and new classes can be added without modifying the rest of the library. The design philosophy mirrors that of OpenAI Gym and OpenSpiel: lightweight abstractions, composability, and ease of extension.

## 2. Creating custom environments

A new discrete-time MFG can be created with a few lines of Python:

```python
from mfglib.env import Environment

env = Environment(
    T=T, S=S, A=A, mu0=mu0, r_max=r_max,
    reward_fn=reward_fn,
    transition_fn=transition_fn
)
```

Users specify:

* `T`: time horizon
* `S`: number of states
* `A`: number of actions
* `mu0`: initial population distribution
* `reward_fn(env, t, L_t)`: returns rewards for time `t` and population distribution `L_t`
* `transition_fn(env, t, L_t)`: returns transition probabilities

MFGLib accepts both function- and class-based callables, enabling advanced use-cases (e.g., contextual environments or stochastic parameters).

## 3. Built-in benchmark environments

MFGLib includes ten pre-implemented environments widely used in the numerical MFG literature, including examples from:

* Perrin (2020)
* Perolat et al. (2021)
* Guo et al. (2019, 2024)
* Cui et al. (2021)

These are available via concise class methods, e.g.:

```python
env = Environment.left_right(T=30, S=50)
```

This allows researchers to evaluate algorithms on a shared set of standardized benchmarks without reimplementing dynamics, facilitating reproducibility and comparison.

## 4. Algorithms and solving MFGs

MFGLib implements several algorithms for computing Nash equilibria in MFGs, including:

* Online Mirror Descent
* Prior Descent
* Fictitious Play
* Occupation Measure Inclusion
* MF-OMO (Mean-Field Optimistic Mirror Optimization)

Using a solver follows a unified interface:

```python
from mfglib.alg import MFOMO

solver = MFOMO(max_iters=200)
policies, expls, runtimes = solver.solve(env)
```

The outputs include:

* the sequence of policy iterates
* exploitability values (a standard metric for deviation from a Nash equilibrium)
* cumulative runtime

MFGLib also supports real-time logging for monitoring convergence.

## 5. Auto-tuning

All solvers include optional automatic hyperparameter tuning. The tuner selects step sizes and algorithmic 
parameters by adaptively evaluating convergence behavior, substantially easing the burden on new users and 
helping practitioners deploy MFG solvers on unfamiliar environments.

# Acknowledgments

An early pre-release version of MFGLib was used internally at Amazon Advertising and influenced several design choices.
We thank the Amazon research teams for feedback and stress-testing the library in production settings. The authors 
would especially like to thank Sareh Nabi, Rabih Salhab, and Lihong Li of Amazon, Xiaoyang Liu of Columbia University, 
and Zhaoran Wang of Northwestern University for their feedback on early versions of MFGLib.

# References
