Introduction
============

Welcome to the QTT framework. This introduction will shortly introduce the framework, and it will guide you through the structure, installation process and how to contribute. We look forward to working with you!

Quantum Technology Toolbox
--------------------------

Quantum Technology Toolbox (QTT) is a Python-based framework developed initialy by `QuTech <https://www.qutech.nl/>`_ for the tuning and calibration of quantum dots and spin qubits.
QuTech is an advanced research center based in Delft, the Netherlands, for quantum computing and quantum internet.
It is a collaboration founded by the Delft University of Technology (`TU Delft <https://www.tudelft.nl/en>`_) and the Netherlands Organisation for Applied Scientific Research (`TNO <https://www.tno.nl/en>`_).

The experiments done on spin-qubits at QuTech make use of the QTT framework to add automated funcionalities and algorithms to their measurement code. 
This paves the way to a more time-efficient, user-friendly and robust code, making more complex research on larger systems possible.
We invite you to use and contribute to QTT. Below we will guide you through the installation.

QTT is the framework on which you can base your measurement and analysis scripts, and QTT itself is based on `Qcodes <https://github.com/qdev-dk/Qcodes>`_. 

 
Installation
------------

QTT is compatible with Python 3.5+.

QTT can be installed as a pip package:

.. code-block:: console

    $ pip install --upgrade qtt 

For development we advice to install from source. First retrieve the source code using git, and then install from the qtt source directory using the command:

.. code-block:: console
   
   $ python setup.py develop

For for Vandersypen research group there are more detailed instructions, read the file `INSTALL-spinqubits.md <https://github.com/VandersypenQutech/spin-projects/blob/master/INSTALL.md>`_ in the spin-projects repository.

Updating QTT
------------

If you registered qtt with Python via ``python setup.py develop`` or ``pip install -e .``, all you need to do to get the latest code is open a terminal window pointing to anywhere inside the repository and run `git pull`.

If you installed qtt via the pip package you can run the pip install comment again:

.. code-block:: console

    $ pip install --upgrade qtt

Usage
-----

In QTT, we use GitHub for combined developing and python for scientific use. If you have some experience with scientific python you will be able to understand the code fairly easily. If not, we urge you to read through some lectures before using the QTT framework. For a general introduction see:

* `Introduction to Github <https://guides.github.com/activities/hello-world/>`_
* `Scientific python lectures <https://github.com/jrjohansson/scientific-python-lectures>`_

We advise to use the following settings when using QTT:

* If you use `Spyder <https://github.com/spyder-ide/spyder>`_ then use the following settings:

  - Use a ``IPython`` console and set the IPython backend graphics option to ``QT``. This ensures correctly displaying the ``ParameterViewer`` and ``DataBrowser``
  - In ``Tools->Preferences->Console->Advanced settings`` uncheck the box ``Enable UMR``

For the usage of algorithms or calibrations we point you to the documentation of those subjects.

Testing
-------

Tests for the qtt packages are contained in the subdirectory ``tests`` and as test functions (``test_*``) in
the code. To run the tests you can run one of the commands below.

.. code-block:: console

    $ pytest

