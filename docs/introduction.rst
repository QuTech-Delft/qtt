Introduction
============

Welcome to the QTT framework. This introduction will shortly introduce the framework, and it will guide you through the structure, installation process and how to contribute. We look forward to working with you!

QuTech Tuning
-------

QuTech Tuning (QTT) is a Python-based framework developed by QuTech for the tuning and calibration of quantum dots and spin qubits. QuTech is an advanced research center based in Delft, the Netherlands, for quantum computing and quantum internet. It is a collaboration founded by the Delft University of Technology (TU Delft) and the Netherlands Organisation for Applied Scientiﬁc Research (TNO).

The experiments done on spin-qubits at QuTech make use of the QTT framework to add automated funcionalities and algorithms to their measurement code. 
This paves the way to a more time-efficient, user-friendly and robust code, making more complex research on larger systems possible.
We invite you to use and contribute to QTT. Below we will guide you through the installation.

Also see

* `QuTech <https://www.qutech.nl/>`_
* `TU Delft <https://www.tudelft.nl/en>`_
* `TNO <https://www.tno.nl/en>`_

Structure
-------

QTT is the framework on which you can base your measurement and analysis scripts, and QTT itself is based on Qcodes. 
In Delft we use a separate repository for the measurement setups (called 'stations'), where each station is programmed in a different way to fit the specific measurements done in that setup. 
In some cases you can use a personal folder in the 'users' repository, but we recommend keeping this to a minimum and always use a common folder with clear descriptions in the code.

* `Qcodes: the basic framework <https://github.com/qdev-dk/Qcodes>`_
* `qtt: contains additional functionality and algorithms <https://github.com/VandersypenQutech/qtt>`_ 
* `spin-projects : Contains hardware descriptions for experimential setups and project data. <https://github.com/VandersypenQutech/spin-projects>`_ 
  
Installation
-------

QTT is compatible with Python 3.5+.

Currently there are no pip packages for qtt. To install download the source code and install using the command:
```
python setup.py develop
```
For for Vandersypen research group there are more detailed instructions, read the file `INSTALL-spinqubits.md<INSTALL-spinqubits.md>`_.

Updating QTT
-------

If you registered qtt with Python via `setup.py develop`, all you need to do to get the latest code is open a terminal window pointing to anywhere inside the repository and run `git pull`

Usage
-------

See the [docs](docs) directory.

For a general introduction also see
* [Introduction to Github](https://guides.github.com/activities/hello-world/)
* [Scientific python lectures](https://github.com/jrjohansson/scientific-python-lectures)

* If you use [Spyder](https://github.com/spyder-ide/spyder) then use the following settings:
- Use a `IPython` console and set the IPyhton backend graphics option to `QT`. This ensures correctly displaying the `ParameterViewer` and `DataBrowser`
- In Tools->Preferences->Console->Advanced settings uncheck the box `Enable UMR`

An example:

.. code:: python

   import qtt
   # load data
   dataset = qtt.data.load_dataset('example')
   # analyse
   results = qtt.algorithms.gatesweep.analyseGateSweep(dataset, fig=100)
   

More examples can be found in the example notebooks.

Contributing
-------

See [Contributing](CONTRIBUTING.md) for information about bug/issue reports, contributing code, style, and testing

Testing
-------

Tests for the qtt packages are contained in the subdirectory `tests` and as test functions (`test_*`) in the code. To run the tests you can run one of the commands below. Note that only `pytest` runs also the tests throughout the code while `python qtt/test.py` only runs the ones in the tests directory.
```
> python qtt/test.py
> pytest # (Windows)
> py.test # (Unix)
```
