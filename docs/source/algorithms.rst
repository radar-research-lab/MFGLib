Algorithms
==========

The following algorithms come pre-implemented with MFGLib.

.. list-table::
   :header-rows: 1

   * - Algorithm
     - Reference
   * - Fictitious Play
     - :cite:t:`perrin2020fictitious`
   * - Online Mirror Descent
     - :cite:t:`perolat2021`
   * - Prior Descent
     - :cite:t:`pmlr-v130-cui21a`
   * - Mean-Field Measure Occupation Measure Optimization
     - :cite:t:`2022:guo`
   * - Occupation Measure Inclusion
     - :cite:t:`2024:MFOMI`

Further algorithm details are provided below. To learn how to implement your own algorithm, see
the section on :ref:`user_defined_algorithms`.

API Reference
-------------

All the pre-implemented algorithms are importable from ``mfglib.alg``.

.. autoclass:: mfglib.alg::FictitiousPlay

.. autoclass:: mfglib.alg::OnlineMirrorDescent

.. autoclass:: mfglib.alg::PriorDescent

.. autoclass:: mfglib.alg::MFOMO

.. autoclass:: mfglib.alg::OccupationMeasureInclusion

.. _user_defined_algorithms:

User-Defined Algorithms
-----------------------

To create an MFGLib-compatible custom algorithm, you can subclass the abstract base class
``mfglib.alg.abc.Algorithm``. Your custom logic will be contained in the ``solve()`` method.

.. autoclass:: mfglib.alg.abc::Algorithm.solve
    :no-index:

Here is a brief example, in which we design an iterative algorithm that applies some abstract
rule :math:`F` to update the policy. To keep the example is simple as possible, we won't allow
initialization with arbitrary policies, we won't implement early stopping, and we won't support
verbose print-outs.

.. code-block:: python

    import typing
    import time

    import torch

    from mfglib.alg.abc import Algorithm
    from mfglib.env import Environment
    from mfglib.scoring import exploitability_score

    class MyAlgorithm(Algorithm):
        def solve(
            env_instance: Environment,
            *,
            pi: typing.Literal["uniform"] | torch.Tensor = "uniform",
            max_iter: int = 100,
            atol: float | None = 0.001,
            rtol: float | None = 0.001,
            verbose: int = 0,
        ) -> tuple[list[torch.Tensor], list[float], list[float]]:
            # Initialize the three lists we will return: policies, exploitability
            # scores, and runtimes.
            pis = []
            expls = []
            rts = []

            # Force uniform initialization. Recall our state and action spaces
            # are, in general, multi-dimensional and must be handled accordingly.
            T = env_instance.T
            S = env_instance.S
            A = env_instance.A
            pi = torch.ones(size=[T + 1, *S, *A]) / torch.tensor(A).prod()

            t0 = time.perf_counter()
            for _ in range(max_iter):
                expl = exploitability_score()
                pis.append(pi)
                expls.append(expl)
                rts.append(time.perf_counter() - t0)
                # F here is some generic policy update rule that, in practice,
                # might take additional arguments.
                pi = F(pi)

            return pis, expls, rts