Measurements
============

To perform measurements several tools are available.

Scan functions
--------------

For basic scanning the following functions are available:

.. autosummary::

    qtt.measurements.scans.scan1D
    qtt.measurements.scans.scan2D
    qtt.measurements.scans.scan2Dfast
    qtt.measurements.scans.scan2Dturbo

For more advances measurements, write your own data acquisition loop.


Parameter viewer
----------------

.. autosummary::
    qtt.gui.parameterviewer.ParameterViewer

VideoMode
---------

.. autosummary::
    qtt.measurements.videomode.VideoMode


Data browser
------------

The data browser can be used to inspect recorded data. 

.. autosummary::
    qtt.gui.dataviewer.DataViewer
    
Live plotting
-------------

Live plotting is done using a `qcodes.QtPlot` window. The window can be setup with

.. autosummary::
    qtt.tools.setupMeasurementWindows


Named gates
-----------


.. autosummary::
    qtt.instrument_drivers.gates.virtual_IVVI



Virtual gates
-------------

.. autosummary::
    qtt.instrument_drivers.virtual_gates.virtual_gates



