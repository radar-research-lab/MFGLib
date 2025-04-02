Algorithms
==========

Built-In
--------

All the algorithms listed below are importable from ``mfglib.alg``.

.. autoclass:: mfglib.alg::FictitiousPlay

.. autoclass:: mfglib.alg::OnlineMirrorDescent

.. autoclass:: mfglib.alg::PriorDescent

.. autoclass:: mfglib.alg::MFOMO

.. autoclass:: mfglib.alg::OccupationMeasureInclusion

User-Defined
------------

To create an MFGLib-compatible custom algorithm, you can subclass the abstract base class
``mfglib.alg.abc.Algorithm``. Your custom logic will be contained in the ``solve()`` method.

.. autoclass:: mfglib.alg.abc::Algorithm.solve
    :no-index:
