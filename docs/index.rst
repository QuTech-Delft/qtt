.. qtt documentation master file, created by
   sphinx-quickstart on Sat Feb  3 15:16:09 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to qtt's documentation!
===============================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


QTT (QuTech Tuning) is a Python package for performing measurements and calibration of spin-qubits.
An example:

.. code:: python

   # imports
   import qtt
   # load data
   dataset = load_data('example')
   # analyse
   results = qtt.algorithms.gatesweep.analyseGateSweep(dataset, fig=100)
   

More examples can be found in the example notebooks.



Documentation
=============

.. toctree::
   :maxdepth: 2
   contributing
   algorithms
   

    
Indices and tables
==================


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

License
=======

.. include:: ../LICENSE.txt


.. include:: ../Contributors.md
