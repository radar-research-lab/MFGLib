Tuning
======

Overview
--------

An algorithm's performance is highly dependent on the choice of hyperparameters. Manually tuning the
hyperparameters for algorithms like ``FictitiousPlay`` or ``OnlineMirrorDescent``, which have only
one tunable parameter, is not very straightforward, let alone for ``PriorDescent`` or ``MFOMO``
which have multiple hyperparameters.

To remedy this, MFGLib algorithms are equipped with automatic hyperparameter tuning based on Optuna
:cite:p:`optuna_2019`, an open-source framework designed for efficient hyperparameter search.

To describe how hyperparameter tuning can increase algorithm performance, we must first define what we mean by
"performance". The MFGLib tuning processes identifies a set of hyperparameters that maximize an objective.
An objective is a function that maps policies, exploitability scores, and runtimes to a real number. In MFGLib,
this objective is represented by the ``Metric`` protocol.

.. note::

    ``Metric`` is a protocol, not a class. This means any object that implements ``evaluate()``
    is a valid ``Metric``. See :ref:`user-defined-ref` for more details.

Given the metric/objective, an algorithm's ``tune()`` method tests various combinations of hyperparameters
and identifies the optimal choice.

.. automethod:: mfglib.alg.abc::Algorithm.tune
    :no-index:

Metrics
-------

MFGLib ships with two built-in metrics and allows for user-defined metrics too.

Built-In
^^^^^^^^

.. autoclass:: mfglib.tuning::FailureRate

.. autoclass:: mfglib.tuning::GeometricMean

.. _user-defined-ref:

User-Defined
^^^^^^^^^^^^

To implement a user-defined method, you simply need to create an object which implements an
``evaluate()`` method with the following signature.

.. autoclass:: mfglib.tuning::Metric.evaluate
    :no-index:
