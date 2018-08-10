.. qtt documentation master file, created by
   sphinx-quickstart on Sat Feb  3 15:16:09 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

QuTech Tuning Documentation
===============================


.. toctree::
   :maxdepth: 2
   :caption: Contents:

Hello, and welcome to the documentation of QuTech Tuning!
   
QuTech Tuning (QTT) is a Python package for performing measurements and calibration of spin-qubits. It is developed at QuTech, an advanced research center for quantum computing and quantum internet. QuTech is a collaboration founded by the Delft University of Technology (`TU Delft <https://www.tudelft.nl/en>`_) and Netherlands Organisation for Applied Scientiﬁc Research (`TNO <https://www.tno.nl/en>`_).

QuTech addresses scientific challenges as well as engineering issues in a joint center of know-how with industrial partners. One of these challenges is to automate the calibration and analysis of measurements pertaining spin qubits. For this purpose, QTT was developed. By sharing this framework with you we hope to work with you on improving it and together forward the development of quantum computers all over the world.

A more elaborate starting guide can be found in the introduction. We do include an example in here to show what QuTech Tuning is capable of:

.. code:: python

   import qtt
   # load data
   dataset = qtt.data.load_dataset('example')
   # analyse
   results = qtt.algorithms.gatesweep.analyseGateSweep(dataset, fig=100)
   

More examples can be found in the example notebooks.


Documentation
=============


.. toctree::
   :maxdepth: 2

   introduction   
   measurements
   calibrations   
   algorithms
   simulation
   contributing
   

    
Indices and tables
==================


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

License
=======

.. include:: ../LICENSE.txt


.. include:: ../Contributors.md
