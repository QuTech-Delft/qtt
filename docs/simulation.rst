Simulation
==========

Simulation code
---------------

The package contains software to simulation quantom dot systems and plot charge stability diagrams. The basic approach is

1. Define parameters of the system (e.g. number of dots, additional energy, etc.)
2. Generate the Hamiltonian from the parameters
3. Calculate the eigenvalues and eigenstatus of the the system
4. Calculate derived properties such as the occupation numbers and charge stability diagrams.

Example Notebooks
----------

* :doc:`Simulate square dot tuning <notebooks/square_dot_tuning>`
* :doc:`Simulate quantum dot systems <notebooks/example_classical_dot_simulation>`
* :doc:`Classical simulation of a triple dot <notebooks/classical_triple_dot>`
* :doc:`Example PAT simulations <notebooks/example_PAT_simulations>`


The virtual dot
---------------

The virtual dot is a simulation model of a linear dot array used for testing code and learning the system.
The simulation is not a good physical simulation, but is sufficient to run some of the measurement and analysis functions.

The documentation is in :py:mod:`qtt.simulation.dotmodel`

