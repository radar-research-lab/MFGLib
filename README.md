<h1 align="center">MFGLib: A Library for Mean-Field Games</h1>

<p align="center">
    <a href="http://mypy-lang.org/"><img alt="Checked with mypy" src="http://www.mypy-lang.org/static/mypy_badge.svg"></a>
    <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
    <a href="https://pycqa.github.io/isort/"><img alt="Imports: isort" src="https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336"></a>
    <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-purple.svg"></a>
</p>

## Usage & Test Environment
API and usage:
* MFGLib consists of two major modules `mfglib/algorithms/` and `mfglib/environments/`. For algorithms, we implement Online Mirror Descent (OMD), Fictitious Play (FP), Prior Descent (PD), and Mean-Field Occupation Measure Optimization (Mf-OMO). For environments, we implement Conservative Treasure Hunting (CTH), Equilibrium Pricing (EP), Left-Right (LR), Rock-Paper-Scissors (RPS), and the SIS epidemic model. 
* Example usage of the APIs: `python experiments.py`. This generates and stores figures of exploitability convergence curves to `figures/`.
* One can also equivalently run  `experiments.ipynb` to understand the usage of the APIs.

We tested the code in a Python 3 anaconda virtual env with PyTorch, NumPy and Matplotlib installed. 

## Minimum working example

Minimum working example code snippet (see `ep_test.py`):

```python
# Required modules
import numpy as np
import matplotlib

matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import time

# Pytorch setup
exec(open('pytorch_settings.py').read())

# environments
from mfglib.environments.EP import EP

# algorithms 
from mfglib.algorithms.prior_descent import Prior_Descent

env_instance = EP()

print('Prior Descent starts ...')
t0 = time.time()
expls_pd, solutions_pd = Prior_Descent(env_instance,
                                       c=0.1,
                                       n_outer=1000,
                                       n_inner=10)
print('Prior Descent finishes in {} seconds\n'.format(time.time() - t0))
print('Start exploitability = {}; Final best exploitability = {}'.format(expls_pd[0], np.min(expls_pd)))

plt.semilogy(np.arange(len(expls_pd)),
             np.array(expls_pd),
             color='r',
             linestyle='-',
             label='PD')

plt.legend(loc=0)
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('Exploitability')
plt.title('EP: Exploitability comparisons among algorithms')
plt.show()
```

## Highlights & Comments
* Prior descent runs GMF-V in each inner loop.
* Multi-dimensional states and actions can be specified in their original shapes in the environemnt.
* T should be 0 for contextual mean-field stage/bandit games. 

## Action Items
* Update `experiments.ipynb`.
* Add printouts for intermediate iterates.
* Add background and API documentation.
* Change the environment interface from tensor outputs for reward and transition given mean-field term to element-wise outputs. 
* Enable jit vectorization and sparse tensors. 
* Enable automatic discretization of the state and action spaces for interval S and A.
* Enable simulator settings and function approximation (e.g., deep learning).
* Enable local constraints like RoAS, global constraints like total revenue lower bound, and global objective like total revenue. Consider penalty/adding to MF-OMO, adding reward/constraint to individual reward, and constrained MFG. Soft vs. hard constraints. Chance constraints can also be considered, and may lead to general utility for the constraint part?
* Can also include proximal terms or constraints on the distance from a target policy (e.g., those coming from some closed-form/theoretical model). Also can consider major-minor MFGs, multi-population MFGs, personalized MFGs, discounted and average reward MFGs, and so on. Check what else? Just make the framework sufficiently flexible. 
* Put r and P into init to remove repeated computation each time the environment is called. 
* Add stopping criteria for the algorithms for early stopping. 
* MF-OMO and Anderson accelerated MF-OMO. Consider redefining the matrices like A_L involved in MF-OMO with the block-wise definitions (and sparse tensors). 
* Add unit tests, especially to ensure that the exploitability, MF-OMO objective and gradient, etc. are correct. One way is to use CVXPY to define the objectives with more explicit mathematical formula for function evaluation checking and use finite differences for gradient evaluation checking. 
* Optional: N-player game P and r as inputs and simulator P and r as inputs?

## Citing
If you wish to cite `MFGLib`, please use the following:
```
@misc{mfglib_code,
    author       = {XXX},
    title        = {{MFGLib}: A Lightweight Library for Mean Field Game Algorithms and Environments, version 0.1.0},
    howpublished = {\url{https://github.com/XXX/MFGLib}},
    year         = {2022}
}
```

