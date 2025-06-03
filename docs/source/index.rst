.. MFGLib documentation master file, created by
   sphinx-quickstart on Mon Mar 27 15:34:25 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

A Library for Mean-Field Games
======================================

.. toctree::
   :maxdepth: 2
   :hidden:

   Quickstart <self>
   algorithms
   environments
   api
   citations
   License <license>

Installation
------------

MFGLib provides pre-built wheels for Python 3.9+ and can be installed via ``pip`` on
all major platforms:

.. code-block::

   $ pip install mfglib

Additionally, the source code is directly available on `GitHub <https://github.com/radar-research-lab/MFGLib>`_.

Your First MFG
--------------
To demonstrate the power of MFGLib, let's use the library to find a Nash equilibrium (NE) solution for an instance
of the **Beach Bar** environment. We begin by importing the ``Environment`` class.

.. jupyter-execute::

   from mfglib.env import Environment

The ``Environment`` class comes equipped with several classmethods which can be used to create instances of environments
well-studied in the mean-field game literature. In addition to the pre-implemented environments, you can also easily implement your
own custom environments. More on pre-implemented and custom environments can be found in the
:ref:`environments:Environments` section.

With ``Environment`` imported, we can then instantiate a **Beach Bar** instance by calling the corresponding classmethod.

.. jupyter-execute::

   beach_bar = Environment.beach_bar()

To "solve" the instance, we must next introduce an algorithm.

.. note::

   Solving an environment means finding an approximate NE solution for it.

In this example, let's use **Online Mirror Descent**. Other options include **Fictitious Play**, **Prior Descent**, and
**Mean-Field Occupation Measure Optimization**. Just like with the environments, MFGLib also supports user-defined algorithms.
More on the algorithms can be found in the :ref:`algorithms:Algorithms` section.

.. jupyter-execute::

   from mfglib.alg import OnlineMirrorDescent

   online_mirror_descent = OnlineMirrorDescent()


Now, we just need to call ``solve()``. The ``solve()`` method returns a three-item tuple: a list of policies (solutions) found
during iteration, exploitability scores of the solutions, and the runtime at each iteration.

.. jupyter-execute::

   solns, expls, runtimes = online_mirror_descent.solve(beach_bar)

The ``solve()`` method allows us to set the initial policy, change the number of iterations, use early stopping, and print the convergence information during iteration. 
More details can be found in the :ref:`api:API Documentation` section.

By default, ``solve()`` runs for 100 iterations and assumes the initial policy to be the uniform policy over the state and action space at each
time step. We can verify this by comparing ``solns[0]`` with ``solns[-1]``

.. jupyter-execute::
	
   solns[0]
   
.. jupyter-execute::
	
   solns[-1]

To compare the two solutions, we look at their **exploitability** scores.

.. jupyter-execute::
	
   expls[0]
   
.. jupyter-execute::
	
   expls[-1]

The computed exploitability score is decreased significantly implying that the last policy is a fairly good
approximation of an NE solution for the **Beach Bar** environment.


You can monitor the progression of an algorithm by setting its ``verbose`` parameter to a positive integer.

.. jupyter-execute::

   _ = online_mirror_descent.solve(beach_bar, verbose=1)

More on Exploitability
^^^^^^^^^^^^^^^^^^^^^^

For most MFGs, we may not be able to get a closed-form of their NE solutions. Therefore, in order to find how close a
given policy is to an NE solution, we use a metric called **exploitability**.

Assume the time horizon of an MFG is :math:`T` and let :math:`\mathcal{T}=\{0, ..., T\}`. Also, denote the state space
by :math:`\mathcal{S}` and the initial state distribution by :math:`\mu_0`. For any policy sequence
:math:`\pi = \{\pi_t\}_{t\in \mathcal{T}}`, let :math:`L=\{L_t\}_{t \in \mathcal{T}}=\Gamma(\pi)` be the induced
mean-field flow. Then, **exploitability** characterizes the sub-optimality of the policy :math:`\pi` under :math:`L` as
follows:

.. math::

   \text{Expl}(\pi) := \max_{\pi'}  V_{\mu_0}^{\pi'}(\Gamma(\pi)) - V_{\mu_0}^\pi(\Gamma(\pi))

where :math:`V_{\mu_0}^{\pi}(L) = \sum_{s \in \mathcal{S}} \mu_0(s)[V_0^\pi(L)]_s`, in which :math:`[V_0^\pi(L)]_s` is
the expected reward starting from state :math:`s` at time :math:`t=0` given the policy :math:`\pi` and mean-field
:math:`L`. In particular, :math:`(\pi, L)` is an NE solution if and only if :math:`L=\Gamma(\pi)` and
:math:`\text{Expl}(\pi)=0`, and a policy :math:`\pi` is an :math:`\epsilon`-NE solution if
:math:`\text{Expl}(\pi) \leq \epsilon`. Refer to :footcite:t:`2022:guo` for a more complete description.


References
^^^^^^^^^^

.. footbibliography::
