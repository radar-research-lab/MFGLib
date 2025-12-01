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

**MFGLib** is an open-source Python library that addresses this gap by providing:

* A modular and extensible API for defining arbitrary discrete-time finite-state MFGs
* Implementations of state-of-the-art algorithms for computing (approximate) Nash equilibria
* A collection of pre-built benchmark environments drawn from the literature
* Tight integration with **Optuna** [@akiba:2019] to provide automatic hyperparameter tuning
* Clear documentation and examples, facilitating both research and industry use

MFGLib includes implementations of several widely used algorithms -- **Online Mirror Descent** 
[@perolat:2021], **Fictitious Play** [@perrin:2020], **MFOMO** [@guo:2023], **MFOMI** [@hu:2024], and **Prior Descent**
[@cui:2021] -- alongside ten benchmark environments drawn from the same references.

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

As a result, researchers re-implement environments, MFG solvers, and population dynamics from scratch—an error-prone 
and time-consuming process that hinders reproducibility and comparison of new algorithms.

**MFGLib addresses this need as the first general-purpose, customizable, and well-documented MFG library.**
It lowers the barrier to entry for new researchers, enables rapid experimentation, and offers practitioners a way to 
prototype MFG-based models without requiring deep expertise in game theory or optimal control.

MFGLib therefore fills an important gap in the scientific software ecosystem for large-population games.

# Acknowledgments

An early pre-release version of MFGLib was used internally at Amazon Advertising and influenced several design choices.
We thank the Amazon research teams for feedback and stress-testing the library in production settings. 

The authors would especially like to thank Sareh Nabi, Rabih Salhab, and Lihong Li of Amazon, Xiaoyang Liu of Columbia 
University, and Zhaoran Wang of Northwestern University for their valuable comments.

# References
