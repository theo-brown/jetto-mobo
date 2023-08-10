Usage
=====

We've designed the package so it is easy to write scripts that run MOBO for a particular JETTO problem.
To do so, you need to:

1. Define your inputs
2. Define your objective functions
3. Write the main evaluation script

There's an example in ``src/jetto_mobo/scripts/ecrh_q_optimisation``; this formed the basis of part of our poster, looking at MOBO of the ECRH input to find good q-profiles.

1. Define inputs
----------------
First, we define the ECRH parameterisation to use.
As ECRH is a plasma profile, we decorate it with ``@jetto_mobo.inputs.plasma_profile``, which tells the optimiser to expect a function of the form ``f(xrho, parameters)``.

.. literalinclude:: ../../src/jetto_mobo/scripts/ecrh_q_optimisation/ecrh_inputs.py
   :language: python
   :linenos:
   :lines: -110

We also need to define the bounds on the parameter values.

.. literalinclude:: ../../src/jetto_mobo/scripts/ecrh_q_optimisation/ecrh_inputs.py
   :language: python
   :linenos:
   :lines: 114-

2. Define objective functions
-----------------------------
Next, we define the objective functions.
For our q-profile optimisation problem, we want to do multi-objective optimisation.
Each of the objectives can be computed from the JETTO profiles and timetraces datasets:

.. literalinclude:: ../../src/jetto_mobo/scripts/ecrh_q_optimisation/q_objectives.py
   :language: python
   :linenos:
   :lines: -60

We can then use ``@jetto_mobo.objectives.objective`` to decorate a function that takes a ``JettoResults`` object and returns the vector of objective values:

.. literalinclude:: ../../src/jetto_mobo/scripts/ecrh_q_optimisation/q_objectives.py
   :language: python
   :linenos:
   :lines: 62-97

If instead we wanted to do single-objective optimisation, we can use ``jetto_mobo.objectives.objective(weights=True)`` to decorate a scalar weighted version of the vector objective function:

.. literalinclude:: ../../src/jetto_mobo/scripts/ecrh_q_optimisation/q_objectives.py
   :language: python
   :linenos:
   :lines: 100-

3. Write the main evaluation script
-----------------------------------

As the evaluation of the objective functions depends on the particular problem at hand, we haven't yet implemented a general framework to run the MOBO loop. (If you think it would be useful, do get in touch!)

Consequently, you'll have to write the evaluation by hand, using our pre-built wrappers.

First, some imports:

.. literalinclude:: ../../src/jetto_mobo/scripts/ecrh_q_optimisation/main.py
   :language: python
   :linenos:
   :lines: -15

3.1 Argument parsing
~~~~~~~~~~~~~~~~~~~~
Because we want it to be easy to use our final script for multiple different runs, we use ``argparse`` to parse the command line arguments:

.. literalinclude:: ../../src/jetto_mobo/scripts/ecrh_q_optimisation/main.py
   :language: python
   :linenos:
   :lines: 16-30

3.2 Evaluation helper function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Next, we define a helper function that takes a set of parameters, creates a JETTO config, sends the config off to be run, and returns the relevant values (input, output, objective) on completion.

.. literalinclude:: ../../src/jetto_mobo/scripts/ecrh_q_optimisation/main.py
   :language: python
   :linenos:
   :lines: 33-101

3.3 Data storage
~~~~~~~~~~~~~~~~
We also define a helper function to save our results to a file. Use your team's preferred data format!

.. literalinclude:: ../../src/jetto_mobo/scripts/ecrh_q_optimisation/main.py
   :language: python
   :linenos:
   :lines: 104-131

3.4 Initialisation
~~~~~~~~~~~~~~~~~~
.. important::
    Parameter bounds must be a tensor! If you initialised them as a numpy array, cast them to a tensor before continuing.
    We do this with:

    .. literalinclude:: ../../src/jetto_mobo/scripts/ecrh_q_optimisation/main.py
        :language: python
        :linenos:
        :lines: 135


Before starting, we need to generate some initial candidates.

.. literalinclude:: ../../src/jetto_mobo/scripts/ecrh_q_optimisation/main.py
   :language: python
   :linenos:
   :lines: 138-182

3.5 Main loop
~~~~~~~~~~~~~
Now we can bring it all together in the main loop.

.. literalinclude:: ../../src/jetto_mobo/scripts/ecrh_q_optimisation/main.py
   :language: python
   :linenos:
   :lines: 184-239
