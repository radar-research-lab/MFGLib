<h1 align="center">MFGLib: A Library for Mean-Field Games</h1>

<p align="center">
    <a href='https://mfglib.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/mfglib/badge/?version=latest' alt='Documentation Status' /></a>
    <a href="http://mypy-lang.org/"><img alt="Checked with mypy" src="http://www.mypy-lang.org/static/mypy_badge.svg"></a>
    <a href="https://github.com/charliermarsh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json&label=linting" alt="Ruff" style="max-width:100%;"></a>
    <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
    <a href="https://pycqa.github.io/isort/"><img alt="Imports: isort" src="https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336"></a>
    <img src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/matteosantama/64b00e45279c946ba2bb77173bf562de/raw/mfglib-covbadge.json" alt="Code coverage badge">
    <a href="https://pypi.org/project/mfglib/"><img src="https://img.shields.io/pypi/v/mfglib" alt="PyPI Version"></a>
    <a href="https://github.com/radar-research-lab/MFGLib/blob/main/LICENSE"><img src="https://img.shields.io/github/license/radar-research-lab/mfglib" alt="License: MIT"></a>
</p>

## Overview
MFGLib is an open-source Python library dedicated to solving Nash equilibria (NEs) for generic mean-field games (MFGs) with a user-friendly and customizable interface, aiming at promoting both applications and research of MFGs. On one hand, it facilitates the creation and analysis of arbitrary user-defined MFG environments with minimal prior knowledge on MFGs. On the other hand, it serves as a modular and extensible code base
for the community to easily prototype and implement new algorithms and environments of MFGs as well as their variants and generalizations.

The official documentation for MFGLib is available at https://mfglib.readthedocs.io/en/latest/. A companion introductory paper can be found at https://arxiv.org/abs/2304.08630. 

## Installation

MFGLib supports all major platforms and can be installed with `pip`:

```
$ pip install mfglib
```

Developers who would like to contribute to the library should refer to the ``Contributing`` section on the 
project's documentation site.

## Usage
Here is an example that shows how to use MFGLib to define an environment called [rock paper scissors](https://mfglib.readthedocs.io/en/latest/environments.html), and solve it using [MF-OMO](https://mfglib.readthedocs.io/en/latest/algorithms.html) with default hyperparameters and tolerances, and plot the exploitability scores over iterations. 

```python
from mfglib.env import Environment
from mfglib.alg import MFOMO
from mfglib.scoring import exploitability_score
import matplotlib.pyplot as plt

# Environment
rock_paper_scissors_instance = Environment.rock_paper_scissors()

# Run the MF-OMO algorithm with default hyperparameters and default tolerances and plot exploitability scores
solns, expls, runtimes = MFOMO().solve(rock_paper_scissors_instance, max_iter=300, verbose=True)

plt.semilogy(runtimes, exploitability_score(rock_paper_scissors_instance, solns)) 
plt.grid(True)
plt.xlabel("Runtime (seconds)")
plt.ylabel("Exploitability")
plt.title("Rock Paper Scissors Environment - MFOMO Algorithm")
plt.show()
```
<img src="https://github.com/radar-research-lab/MFGLib/blob/main/images/mwe.png" alt= “” width=800 height=600>

Please refer to the [documentation](https://mfglib.readthedocs.io/en/latest/) for more info on how to use the library. 

## Citing
If you wish to cite MFGLib, please use the following:
```
@article{mfglib,
    author  = {Guo, X. and Hu, A. and Santamaria, M. and Tajrobehkar, M. and Zhang, J.},
    title   = {{MFGLib}: A Library for Mean Field Games},
    journal = {arXiv preprint arXiv:2304.08630},
    year    = {2023}
}
```
If you find MFGLib to be helpful and would like to send virtual kudos, please consider leaving a star on the [GitHub repository](https://github.com/radar-research-lab/MFGLib).

