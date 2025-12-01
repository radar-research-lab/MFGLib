---
title: "MFGLib: A Library for Mean-Field Games"
tags:
    - Mean-Field Games
    - Nash Equilibrium
    - Python
    - Large Population Games
    - Multi-Agent Systems
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
and the population distribution [@lasry:2007]. It has been shown that the Nash equilibrium policy
of the mean-field game is an $\epsilon$-Nash equilibrium of the $N$-player game with $\epsilon = O(1 / \sqrt{N})$ 
[@huang:2006] and in practice even games with small $N$ on the order of tens can be well-approximated by MFGs 
[@guo:2019, @kizilkale:2019, @cabannes:2021]. Due to their tractability, MFG models have become widely used in 
applications such as digital advertising, high-frequency trading, dynamic pricing, transportation, and behavioral modeling.

Despite the rapid growth of the MFG literature, researchers and practitioners lack a unified, 
open-source software package for defining and solving their own MFG problems. Existing MFG 
implementations are largely one-off, built for experiments within individual papers, and are 
not designed for general use or extensibility.

**MFGLib** is an open-source Python library that addresses this gap by providing:

* A modular and extensible API for defining arbitrary discrete-time finite-state MFGs
* Implementations of state-of-the-art algorithms for computing (approximate) Nash equilibria
* A collection of pre-built benchmark environments drawn from the literature
* Tight integration with **Optuna** [@akiba:2019] to provide automatic hyperparameter tuning
* Clear documentation and examples, facilitating both research and industry use

The library is implemented in Python, maintained on GitHub, and can be installed via `pip install mfglib`. Full 
documentation, tutorials, and example notebooks are available at https://mfglib.readthedocs.io/en/latest/.

# Statement of need

Over the past decade, numerical methods for MFGs have attracted growing interest across control theory, economics, 
reinforcement learning, and operations research. However, the field lacks a standardized, user-friendly software tool 
analogous to **OpenSpiel** [@lanctot:2019] for general games or **CVXPY** [@diamond:2016] for convex optimization.

Existing tools fall short for one of two reasons:

1. General game libraries such as **Nashpy** [@knight:2018] and **QuantEcon** [@batista:2024] do not provide adequate 
support for mean-field interaction structures, population distributions, or MFG-specific solution methods.

2. MFG-specific repositories such as **gmfg-learning** [@cui:2022] or **entropic-mfg** [@benamou:2019] are designed to 
reproduce experiments from individual publications. They lack reusable abstractions, extensible environment definitions, 
and stable APIs.

**OpenSpiel** [@lanctot:2019] includes an MFG module, but it lacks customizability and a user-friendly
API for general users. According to its documentation, the MFG code is experimental and recommended only
for internal use.

As a result, researchers re-implement environments, MFG solvers, and population dynamics from scratch—an
error-prone and time-consuming process that hinders reproducibility and comparison of new algorithms.

**MFGLib addresses this need as the first general-purpose, customizable, and well-documented MFG library.**
It lowers the barrier to entry for new researchers, enables rapid experimentation, and offers practitioners
a way to prototype MFG-based models without requiring deep expertise in game theory or optimal control.

# Key Features

## User-Friendly Environment Creation

Users can define custom MFG environments by providing reward functions, transition functions, and basic 
problem parameters:

```python
from mfglib.env import Environment

user_env = Environment(
    T=T, S=S, A=A, mu0=mu0, r_max=r_max,
    reward_fn=reward_fn,
    transition_fn=transition_fn
)
```

The reward and transition functions are simple callables with straightforward signatures:

```python
def reward_fn(env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor: ...
def transition_fn(env: Environment, t: int, L_t: torch.Tensor) -> torch.Tensor: ...
```

This design allows users to create environments with minimal code while maintaining mathematical clarity. 
The library also includes ten pre-loaded benchmark environments that can be instantiated directly:

```python
env = Environment.left_right()
env = Environment.beach_bar()
env = Environment.building_evacuation(n_floors=4, floor_l=3, floor_w=3)
```

## State-of-the-Art Algorithms

MFGLib includes implementations of five widely used algorithms: **Online Mirror Descent** [@perolat:2021], 
**Fictitious Play** [@perrin:2020], **MFOMO** [@guo:2023], **MFOMI** [@hu:2024], and **Prior Descent** 
[@cui:2021]. These algorithms encompass many other existing methods as special cases, such as fixed point 
iteration and GMF-V algorithms [@guo:2019].

Once an environment is created, users can solve it with any implemented algorithm:

```python
from mfglib.alg import MFOMO

alg = MFOMO(**kwargs)
policies, expls, runtimes = alg.solve(env)
```

The `solve` method returns policy iterates, exploitability scores (which evaluate closeness to Nash 
equilibrium), and cumulative runtimes. A formatted log of the iteration process is optionally printed 
in real-time to help users monitor performance.

## Automatic Hyperparameter Tuning

Every algorithm requires hyperparameters that can drastically influence convergence properties. MFGLib 
provides a built-in tuner based on Optuna [@akiba:2019] to automatically select optimal hyperparameters:

```python
from mfglib.tuning import GeometricMean

tune_result = mfomo.tune(
    metric=GeometricMean(),
    envs=env_suite
)
mfomo_tuned = MFOMO.from_study(tune_result)
policies, expls, runtimes = mfomo_tuned.solve(env)
```

The tuner can optimize across single instances or environment suites with multiple policy initializations 
and customizable metrics. Users can also implement their own metrics with minimal effort.

## High-Dimensional State and Action Spaces

MFGLib uses PyTorch tensors to represent policies, mean-fields, and rewards while preserving the original 
structure of state and action spaces. Rather than flattening high-dimensional spaces into one-dimensional 
representations, the library maintains the natural structure, providing higher interpretability and more 
flexible user interactions.

## Code Quality and Accessibility

MFGLib adheres to high standards of code quality through:

* Continuous integration with comprehensive unit testing
* Code formatting with `black` and `ruff`
* Strict type-checking with `mypy` to eliminate type-safety bugs
* Pure Python implementation with no complicated build process
* Simple, navigable folder structure

The library is easily accessible to anyone familiar with Python and welcomes outside contributors. 

# Acknowledgments

An early pre-release version of MFGLib was used internally at Amazon Advertising and influenced several design choices.
We thank the Amazon research teams for feedback and stress-testing the library in production settings. 

The authors would especially like to thank Sareh Nabi, Rabih Salhab, and Lihong Li of Amazon, Xiaoyang Liu of Columbia 
University, and Zhaoran Wang of Northwestern University for their valuable comments.

# References
