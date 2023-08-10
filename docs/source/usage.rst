Usage
=====

We've designed the package so it is easy to write scripts that run MOBO for a particular JETTO problem.
To do so, you need to:

1. Define your inputs
2. Define your objective functions
3. Write the main evaluation script

In this section, we run through the ``ecrh_q_optimisation`` example from `<https://github.com/theo-brown/jetto-mobo/tree/main/examples/ecrh_q_optimisation>`_; this formed the basis of our :doc:`SOFE2023 poster <publications>`, looking at MOBO of the ECRH input to find good q-profiles.

1. Define inputs
----------------
First, we define the ECRH parameterisation to use.
As ECRH is a plasma profile, we decorate it with ``@jetto_mobo.inputs.plasma_profile``, which tells the optimiser to expect a function of the form ``f(xrho, parameters)``.

.. literalinclude:: ../../examples/ecrh_q_optimisation/ecrh_inputs.py
   :language: python
   :caption: `examples/ecrh_q_optimisation/ecrh_inputs.py <https://github.com/theo-brown/jetto-mobo/tree/main/examples/ecrh_q_optimisation/ecrh_inputs.py>`__
   :linenos:
   :lineno-start: 1
   :lines: -110

We also need to define the bounds on the parameter values.

.. literalinclude:: ../../examples/ecrh_q_optimisation/ecrh_inputs.py
   :language: python
   :caption: `examples/ecrh_q_optimisation/ecrh_inputs.py <https://github.com/theo-brown/jetto-mobo/tree/main/examples/ecrh_q_optimisation/ecrh_inputs.py>`__
   :linenos:
   :lineno-start: 114
   :lines: 114-

1. Define objective functions
-----------------------------
Next, we define the objective functions.
For our q-profile optimisation problem, we want to do multi-objective optimisation.
Each of the objectives can be computed from the JETTO profiles and timetraces datasets:

.. literalinclude:: ../../examples/ecrh_q_optimisation/q_objectives.py
   :language: python
   :caption: `examples/ecrh_q_optimisation/q_objectives.py <https://github.com/theo-brown/jetto-mobo/tree/main/examples/ecrh_q_optimisation/q_objectives.py>`__
   :linenos:
   :lineno-start: 1
   :lines: -60

We can then use ``@jetto_mobo.objectives.objective`` to decorate a function that takes a ``JettoResults`` object and returns the vector of objective values:

.. literalinclude:: ../../examples/ecrh_q_optimisation/q_objectives.py
   :language: python
   :caption: `examples/ecrh_q_optimisation/q_objectives.py <https://github.com/theo-brown/jetto-mobo/tree/main/examples/ecrh_q_optimisation/q_objectives.py>`__
   :linenos:
   :lineno-start: 62
   :lines: 62-97

If instead we wanted to do single-objective optimisation, we can use ``jetto_mobo.objectives.objective(weights=True)`` to decorate a scalar weighted version of the vector objective function:

.. literalinclude:: ../../examples/ecrh_q_optimisation/q_objectives.py
   :language: python
   :caption: `examples/ecrh_q_optimisation/q_objectives.py <https://github.com/theo-brown/jetto-mobo/tree/main/examples/ecrh_q_optimisation/q_objectives.py>`__
   :linenos:
   :lineno-start: 100
   :lines: 100-

3. Write the main evaluation script
-----------------------------------

As the evaluation of the objective functions depends on the particular problem at hand, we haven't yet implemented a general framework to run the MOBO loop. (If you think it would be useful, do get in touch!)

Consequently, you'll have to write the evaluation by hand, using our pre-built wrappers.

3.1 Evaluation helper function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We define a helper function that takes a set of parameters, creates a JETTO config, sends the config off to be run, and returns the relevant values (input, output, objective) on completion.

.. literalinclude:: ../../examples/ecrh_q_optimisation/evaluation.py
   :language: python
   :caption: `examples/ecrh_q_optimisation/evaluation.py <https://github.com/theo-brown/jetto-mobo/tree/main/examples/ecrh_q_optimisation/evaluation.py>`__
   :linenos:
   :lineno-start: 1
   :lines: -88

3.2 Data storage
~~~~~~~~~~~~~~~~
We also define a helper function to save our results to a file. Use your team's preferred data format!

.. literalinclude:: ../../examples/ecrh_q_optimisation/evaluation.py
   :language: python
   :caption: `examples/ecrh_q_optimisation/evaluation.py <https://github.com/theo-brown/jetto-mobo/tree/main/examples/ecrh_q_optimisation/evaluation.py>`__
   :linenos:
   :lineno-start: 91
   :lines: 91-


3.3 Argument parsing
~~~~~~~~~~~~~~~~~~~~
Because we want it to be easy to use our final script for multiple different runs, we use ``argparse`` to parse the command line arguments:

.. literalinclude:: ../../examples/ecrh_q_optimisation/main.py
   :language: python
   :caption: `examples/ecrh_q_optimisation/main.py <https://github.com/theo-brown/jetto-mobo/tree/main/examples/ecrh_q_optimisation/main.py>`__
   :linenos:
   :lineno-start: 1
   :lines: -27


3.4 Initialisation
~~~~~~~~~~~~~~~~~~
.. important::
    Parameter bounds must be a tensor! If you initialised them as a numpy array, cast them to a tensor before continuing.
    We do this with:

    .. literalinclude:: ../../examples/ecrh_q_optimisation/main.py
        :language: python
        :caption: `examples/ecrh_q_optimisation/main.py <https://github.com/theo-brown/jetto-mobo/tree/main/examples/ecrh_q_optimisation/main.py>`__
        :linenos:
        :lineno-start: 30
        :lines: 30


Before starting, we need to generate some initial candidates.

.. literalinclude:: ../../examples/ecrh_q_optimisation/main.py
   :language: python
   :caption: `examples/ecrh_q_optimisation/main.py <https://github.com/theo-brown/jetto-mobo/tree/main/examples/ecrh_q_optimisation/main.py>`__
   :linenos:
   :lineno-start: 32
   :lines: 32-81

3.5 Main loop
~~~~~~~~~~~~~
Now we can bring it all together in the main loop.

.. literalinclude:: ../../examples/ecrh_q_optimisation/main.py
   :language: python
   :caption: `examples/ecrh_q_optimisation/main.py <https://github.com/theo-brown/jetto-mobo/tree/main/examples/ecrh_q_optimisation/main.py>`__
   :linenos:
   :lineno-start: 83
   :lines: 83-
