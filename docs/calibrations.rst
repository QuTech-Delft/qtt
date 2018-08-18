Calibrations
============

This document contains some guidelines for creating robust and re-usable calibration and tuning routines.

1. Separate your measurement, analysis and visualization part of the code
2. Each calibration result should be a dictionary with the calibration results. A couple of fields have a special meaning:

   - ``type``: Contains a string with the type of calibration. E.g. T1 or pinchoff
   - ``description``: A string with a more elaborate description of the calibration procedure
   - ``dataset``: Contains either the measurement data or a reference (string) to the measurement data
   - ``status``: A string that can be "failed", "good" or "unknown"

   The calibration framework will automatically add the following fields to the dictionary:

   - ``timestamp`` (string): String with the date and time of calibration
   - ``tag`` (string): Contains identifier of the calibration

3. The calibration results are stored in a central database. The calibration results are identified by tags which are lists of strings, e.g. ``['calibration', 'qubit1', 'T1']``.

An example of a calibration result


.. code-block:: python
    
    # measure a pinchoff-scan
    ...

    # analyse the scan
    $ result = analyseGateSweep(dataset)
    analyseGateSweep: leftval 0.0, rightval 0.3
    $ print(result)

    {'_mp': 392,
     '_pinchvalueX': -450.0,
     'dataset': '2018-08-18/16-33-50_qtt_generic',
     'description': 'pinchoff analysis',
     'goodgate': True,
     'highvalue': 0.9999999999999998,
     'lowvalue': 9.445888759986548e-18,
     'midpoint': -408.0,
     'midvalue': 0.29999999999999993,
     'pinchvalue': -458.0,
     'type': 'gatesweep',
     'xlabel': 'Sweep plunger [mV]'}
     
   
    
Storage
-------

For storage we recommend to use HDF5.