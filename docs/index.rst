.. qtt documentation master file, created by
   sphinx-quickstart on Sat Feb  3 15:16:09 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Hello, and welcome to the documentation of QTT!
===============================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


QTT (QuTech Tuning) is a Python package for performing measurements and calibration of spin-qubits. 
It is developed at QuTech, QuTech is an advanced research center for quantum computing and quantum internet, a collaboration founded by TU Delft and TNO, the Dutch Research Centre for Applied Sciences Research.
QuTech addresses scientific challenges as well as engineering issues in a joint center of know-how with industrial partners. One of these challenges is to automate parts of the calibration and analysis of spin-qubits. 
By sharing this code with you we hope to forward the development of quantum computers all over the world and work together with you to improve the code further.



An example:

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
