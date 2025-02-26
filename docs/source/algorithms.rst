Algorithms
==========

Overview
--------

This library comes with five implemented MFG algorithms:


* Fictitious Play (:footcite:t:`perrin2020fictitious`)
* Online Mirror Descent (:footcite:t:`perolat2021`)
* Prior Descent (:footcite:t:`pmlr-v130-cui21a`)
* MF-OMO (:footcite:t:`2022:guo`)
* MF-OMI-FBS (:footcite:t:`2024:MFOMI`)

Creating custom algorithms is as simple as subclassing the abstract class ``mfglib.alg.abc.Algorithm``.

In what follows, we provide more information on each of these algorithms. The default instance of the
**Rock Paper Scissors** environment is considered to illustrate algorithms performance.

Fictitious Play
^^^^^^^^^^^^^^^

For detailed explanation of the algorithm, you can refer to :footcite:t:`perrin2020fictitious`. The implementation of
**Fictitious Play** in this library is based on **Fictitious Play Damped** introduced in :footcite:t:`perolat2021`.
The damped version generalizes the original algorithm by adding a learning rate parameter :math:`\alpha`. It is worth mentioning that the
**Fixed Point Iteration** algorithm is a special case of **Fictitious Play Damped** with :math:`\alpha=1`.

**Hyperparameters:**

* ``alpha``: The learning rate. It can be any number in :math:`[0, 1]` or ``None``. If ``None``, the implementation would be inline with the original **Fictitious Play** algorithm as the learning rate used in the :math:`n^{\mathrm{th}}` iteration would be :math:`\frac{1}{n+1}`. The default is ``None``.

You can create an instance by importing and instantiating ``FictitiousPlay``. Let's see this algorithm in action.

.. plot:: plots.py plot_fictitious_play
    :show-source-link: False

Online Mirror Descent
^^^^^^^^^^^^^^^^^^^^^

You can refer to :footcite:t:`perolat2021` for a detailed explanation.

**Hyperparameters:**

* ``alpha``: The learning rate. The default is 1.0.

Let's see how **Online Mirror Descent** performs.

.. plot:: plots.py plot_online_mirror_descent
    :show-source-link: False

Prior Descent
^^^^^^^^^^^^^

The algorithm is explained in detail by :footcite:t:`pmlr-v130-cui21a`.

**Hyperparameters:**

* ``eta``: The temperature. It can be any positive number. The default is 1.0.
* ``n_inner``: Determines the number of iterations between prior updates. It can be a positive integer or ``None``. If ``None``, prior remains intact, which is basically the **GMF-V** algorithm :footcite:t:`guo2021learning`.

You can create an instance by importing and instantiating ``PriorDescent``. Let's give it a run.

.. plot:: plots.py plot_prior_descent
    :show-source-link: False

MF-OMO
^^^^^^

MF-OMO, or Mean-Field Occupation Measure Optimization, is introduced and explained in detail
in :footcite:t:`2022:guo`.

**Hyperparameters:**

**MFOMO** has several hyperparameters, each of which plays an important role in the algorithm performance. Below,
only one of them is described. Detailed information on the rest of the hyperparameters is provided in the :ref:`algorithms:More on MFOMO` subsection.

* ``optimizer``: Determines the optimization algorithm and its configuration. ``optimizer`` should be a dictionary with two keys:

  1.   ``"name"``: The name of a PyTorch optimizer, e.g., ``Adam``, ``SGD``, ``RMSprop``, etc.
  2.   ``"config"``: The desired configuration for the selected optimizer. For example, if we choose ``Adam``, then we can set the value of ``"config"`` as ``{"lr": 0.1, "amsgrad": True}``.

  By default, it is set to ``{"name": "Adam", "config": {"lr": 0.1}}``.
  
You can create an instance by importing and instantiating ``MFOMO``. Let's see this algorithm in action.

.. plot:: plots.py plot_mf_omo
    :show-source-link: False

MF-OMI-FBS
^^^^^^^^^^

MF-OMI-FBS, short for Mean-Field Occupation Measure Inclusion with Forward-Backward Splitting, is introduced in :footcite:t:`2024:MFOMI`.

.. autoclass:: mfglib.alg::OccupationMeasureInclusion
    :no-index:
    :exclude-members: __new__

.. jupyter-execute::
    :hide-code:

    import matplotlib.pyplot as plt

    from mfglib.env import Environment
    from mfglib.alg import OccupationMeasureInclusion

    env = Environment.rock_paper_scissors()
    for alpha in [0.01, 0.05]:
        for eta in [0.0, 0.2]:
            _, expls, _ = OccupationMeasureInclusion(alpha=alpha, eta=eta).solve(
                env, max_iter=300
            )
            plt.semilogy(expls, label=f"{alpha=}, {eta=}")

    plt.legend()
    plt.grid()
    plt.xlabel("Iteration")
    plt.ylabel("Exploitability")
    plt.title("MF-OMI-FBS on Rock Paper Scissors environment");

Tuning
------
As you may have noticed in the previous section, choosing the right set hyperparameters is essential to get the best
performance out of an algorithm. A set of hyperparameters could work for one environment but result in a poor
performance in other environments. Even two distinct instances of the same environment could require very different
sets of hyperparameters. Accordingly, manually tuning the hyperparameters for algorithms such as ``FictitiousPlay``
and ``OnlineMirrorDescent``, despite having only one tunable parameter, is not very straightforward, let alone for
``PriorDescent`` or ``MFOMO`` which have several hyperparameters.

All the algorithms in ``MFGLib`` are equipped with built-in hyperparameter tuning which can be used to tune the
algorithms. Under the hood, the tuners are based on Optuna (:footcite:t:`optuna_2019`), an open-source  optimization
framework used to automate hyperparameter search. The tuning procedure aims to minimize some measure of performance.
In ``MFGLib``, such a measure is represented by a ``Metric`` object.

.. autoclass:: mfglib.tuning::Metric
    :members:

To design a custom ``Metric``, a user need only to create a class that implements ``.evaluate()``. ``MFGLib``
comes equipped with two built-in metrics.

.. autoclass:: mfglib.tuning::FailureRate
    :class-doc-from: init

.. autoclass:: mfglib.tuning::GeometricMean
    :class-doc-from: init


An algorithm's ``.tune()`` takes several other additional parameters:

.. automethod:: mfglib.alg.abc::Algorithm.tune

To demonstrate how the tuner works, let's consider an instance of the ``BuildingEvacuation`` environment and tune
the ``OnlineMirrorDescent`` algorithm on it. We will compare the performance of the tuned and default algorithms.

The tuner runs for 20 trials and the time limit is 60 seconds. Note that
``envs=[Environment.building_evacuation(T=5, n_floor=10, floor_l=5, floor_w=5)]`` as we want to tune the algorithm only
on one specific environment instance.

.. plot::
    :context:
    :nofigs:
    :show-source-link: False

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

.. plot::
    :context:
    :nofigs:
    :include-source:
    :show-source-link: False

    from mfglib.alg import OnlineMirrorDescent
    from mfglib.env import Environment
    from mfglib.tuning import GeometricMean

    env = Environment.building_evacuation(
        T=5, n_floor=10, floor_l=5, floor_w=5
    )
    omd = OnlineMirrorDescent()

    # Compute exploitability scores with default algorithm parameters.
    _, expls_default, _ = omd.solve(env)

    # Tune the algorithm to create an `optuna.Study` object.
    optuna_study = omd.tune(
        metric=GeometricMean(stat="expl"), envs=[env], n_trials=20
    )

    # Modify the algorithm's parameters in-place.
    for key, val in optuna_study.best_params.items():
        setattr(omd, key, val)

    # Re-run the tuned algorithm on the environment.
    _, expls_tuned, _ = omd.solve(env)

.. plot::
    :context:
    :show-source-link: False

    plt.figure(figsize=(10, 5))
    plt.title("Exploitability Scores")
    plt.semilogy(expls_default, label="expls_default")
    plt.semilogy(expls_tuned, label="expls_tuned")
    plt.legend(loc=3)
    plt.grid()
    plt.xlabel("iteration")
    plt.show()


The tuned algorithm outperforms the default. By setting lower exploitability thresholds, we might get even a better
performance, but we may need to run the tuner for more trials and time.

A few remarks about the tuner:

1. To ensure that the tuned hyperparameters work well on a broader range of environments, we can pass a list of multiple environment instances to the tuner via the ``envs`` input argument.
2. The default set of hyperparameters may not be used during the tuning process. Consequently, there might be cases in which the default algorithm outperforms the tuned algorithm.
3. Depending on the the values of the tuner's inputs such as ``max_iter``, ``atol``, ``rtol``, etc., it is possible that none of the algorithm trials solve any of the environment instances in which case the tuner does nothing.

More on MFOMO
-------------

**MFOMO** reformulates the problem of finding the NE solutions of an MFG as an optimization problem. To be precise, finding an NE solution of an MFG is equivalent to solving the following constrained optimization problem:

.. math::
   :label: mfomo

   \text{minimize}_{L, z, y} \quad &||A_LL-b||_2^2 + ||A_L^Ty + z - c_L||_2^2 + z^TL\\
   \text{subject to} \quad &L\geq 0, ~ 1^TL_t=1 ~ \forall t \in \{0, ..., T\},\\
   &1^Tz \leq SA(T^2+T+2)r_{\max},\\
   &||y||_2\leq \frac{S(T+1)(T+2)}{2}r_{\max}.

For detailed description of the variables and parameters in this optimization formulation, please refer to :footcite:t:`2022:guo`.

We can solve :eq:`mfomo` using an optimization algorithm such as **Projected Gradient Descent**. Furthermore, :footcite:t:`2022:guo` suggests several techniques using which could result in an improvment in the convergence. 
By modifying the corresponding input parameters, you can change the optimizer and/or enable different techniques to be used while running **MFOMO**. 
In what follows, we categorize the algorithm hyperparameters based on their usage and describe them.


**Optimization Algorithm:** You can run **MFOMO** with different optimization algorithms. Currently, we only support PyTorch optimizers. The following input argument allows you to set your desired optimizer:

*   ``optimizer``: Determines the optimization algorithm and its configuration. ``optimizer`` should be a dictionary with two keys:

  1.   ``"name"``: The name of a PyTorch optimizer, e.g., ``Adam``, ``SGD``, ``RMSprop``, etc. 
  2.   ``"config"``: The desired configuration for the selected optimizer. For example, if we choose ``Adam``, then we can set the value of ``"config"`` as ``{"lr": 0.1, "amsgrad":True}``. 
  
  By default, ``optimizer`` is set to ``{"name": "Adam", "config": {"lr": 0.1}}``.

.. note::

   Since we are solving a constrained optimization problem, the optimizer iterations are automatically projected onto the constraint set after each iteration. 

**Parameterized Formulation:** Replacing the constrained optimization problem :eq:`mfomo` with a smooth unconstrained problem enables us to use a broader range of optimization solvers. As explained in Appendix A.3 of :footcite:t:`2022:guo`, 
we can reparameterize the variables in **MFOMO** to completely get rid of the constraints. The new problem is called the "parameterized" formulation. Using the following input argument, 
we can switch between the different formulations:

*   ``parameterize``: Optionally solve the alternate "parameterized" formulation. Default is ``False``.  

**Hat Initialization:**

*   ``hat_init``: Determines whether to use "hat initialization", in which, given the initial mean-field :math:`L`, initial :math:`z, y` are set to :math:`\hat{z}(L), \hat{y}(L)` as explained in Proposition 6 of :footcite:t:`2022:guo`. Default is ``False``. 

**Redesigned Objective:** In the optimization problem :eq:`mfomo`, one can assign different coefficients to the three terms in the objective function, and come up with a "redesigned objective". To be precise, the redesigned objective is
:math:`c_1 ||A_LL-b||_2^2 + c_2 ||A_L^Ty + z - c_L||_2^2 + c_3 z^TL.`

We can also apply different norms (L1 or L2) to the objective terms. The following input arguments determine the parameters of the redesigned objective:

*   ``c1``, ``c2``, ``c3``: The redesigned objective coefficients. Default is 1 for all the coefficients. 
 
.. note::

   Without loss of generality, we can always let ``c3=1``.
   
*   ``loss``: Determines the type of norm (L1, L2, or both) used in the redesigned objective function. Three available options are listed below:

  1.   ``"l1"``: The objective will be :math:`c_1 ||A_LL-b||_1 + c_2 ||A_L^Ty + z - c_L||_1 + c_3 z^TL`.
  2.   ``"l2"``: The objective will be :math:`c_1 ||A_LL-b||_2^2 + c_2 ||A_L^Ty + z - c_L||_2^2 + c_3 (z^TL)^2`.
  3.   ``"l1_l2"``: The objective will be :math:`c_1 ||A_LL-b||_2^2 + c_2 ||A_L^Ty + z - c_L||_2^2 + c_3 z^TL`. 

  The default is ``"l1_l2"``.

**Adaptive Residual Balancing:** We can adaptively change the coefficients (:math:`c1`, :math:`c2`, and :math:`c3`) of the redesigned objective based on the value of their corresponding objective term. This can be done using the following input arguments:

*   ``m1``, ``m2``, ``m3``: Determine the parameters used for adaptive residual balancing. 

  Let's denote by :math:`O_1` the value of the first objective term (depending on the norm used, it could be either :math:`||A_LL-b||_1` or :math:`||A_LL-b||_2^2`), and let :math:`O_2` and :math:`O_3` be the values of the second and third objective terms, respecively. 
  When adaptive residual balancing is applied, we modify the coefficients in the followin cases:

    1. If :math:`O_1/ \max(O_2, O_3) > m_1`, then multiply ``c1`` by ``m2``. 
    2. If :math:`O_1/ \min(O_2, O_3) < m_3`, then divide ``c1`` by ``m2``. 
    3. If :math:`O_2/ \max(O_1, O_3) > m_1`, then multiply ``c2`` by ``m2``. 
    4. If :math:`O_2/ \max(O_1, O_3) > m_3`, then divide ``c2`` by ``m2``. 
    
    
*   ``rb_freq``: Determines the frequency of residual balancing. It can be a positive integer or ``None``. If ``None``, residual balancing will not be applied.

**Initialization:** We can set the initial policy for any algorithm using the input argument ``pi`` through the ``solve()`` method. 
**MFOMO** uses the initial policy to compute the initial values of the variables :math:`L`, :math:`z`, and :math:`y`. However, if you want to initialize these variables directly, you can do so using the following input arguments:

*   ``L``, ``z``, ``y``: The initial values of math:`L`, :math:`z`, and :math:`y`. Default is ``None``. If not ``None``, these values overwrite the initial values derived from the initial policy. 

*   ``u``, ``v``, ``w``: The initial values of the variables math:`u`, :math:`v`, and :math:`w` used in the "parameterized" formulation. Refer to the Appendix A.3 of :footcite:t:`2022:guo` for more information. Default is ``None``. If not ``None``, these values overwrite the initial values derived from the initial policy. 


References
^^^^^^^^^^

.. footbibliography::