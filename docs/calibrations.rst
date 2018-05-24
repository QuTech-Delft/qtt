Calibrations
============

This document contains some guidelines for creating robust and re-usable calibration and tuning routines.

1. Separate your measurement, analysis and visualization part of the code
2. Each calibration result should be a dictionary with the calibration results. A couple of fields have a special meaning:
    * type: Contains a string with the type of calibration. E.g. T1 or pinchoff
    * description: A string with a more elaborate description of the calibration procedure
    * dataset: Contains either the measurement data or a reference (string) to the measurement data
    * status: A string that can be "failed", "good" or "unknown"
   The calibration framework will automatically add the following fields to the dictionary:
    * timestamp: String with the date and time of calibration
    * status:
3. The calibration results are stored in a central database. The calibration results are identified by tags which are
 lists of strings, e.g. `['calibration', 'qubit1', 'T1']`.

