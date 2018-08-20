Simulation
==========

Simulation code
---------------

The package contains software to simulation quantom dot systems and plot charge stability diagrams. The basic approach is

1. Define parameters of the system (e.g. number of dots, additional energy, etc.)
2. Generate the Hamiltonian from the parameters
3. Calculate the eigenvalues and eigenstatus of the the system
4. Calculate derived properties such as the occupation numbers and charge stability diagrams.

Example notebooks: `Classical dot simulation <classicaldot_>`__, `2x2 dot system <twoxtwo>`_.

.. _twoxtwo: notebooks/square_dot_tuning.html
.. _classicaldot: notebooks/example_classical_dot_simulation.html

The documentation is in :py:mod:`qtt.simulation.dotsystem`



The virtual dot
---------------

The virtual dot is a simulation model of a linear dot array used for testing code and learning the system.
The simulation is not a good physical simulation, but is sufficient to run some of the measurement and analysis functions.


For a complete example, see the notebook `Using the virtual dot array <notebooks/example_virtual_dot_array.html>`_.

The documentation is in :py:mod:`qtt.simulation.virtual_dot_array`
