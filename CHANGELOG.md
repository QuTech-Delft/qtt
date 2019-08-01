# Changelog
All notable changes to the Quantum Tuning Toolkit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Added
- Added boxcar filter to algorithms (#597).
- Added a changelog to QTT (#591).
- Added load example dataset method (#586).
- Added chirp signal to QuPulse templates (#582).
- Made QTT python 3.7 compatible and Travis now runs a python 3.7 build (#578).
- Allow json files in the DataBrowser (#557).
- Added plot_dataset from spin-projects repo (#565).
- New post processing functionality is added (#563, #564).
- Added a lock-in amplifier interface for the Zurich instruments UHFLI (#560).
- A JSON encoding for qcodes.DataSet is added to the serialized (#558).
- Added scope reader for the Zurich Instruments UHFLI. (#554, #980).
- Add option to compensate for re-arm time of M4i trigger.
- Add methods for performing operations on datasets. (#593)
- The new virtual AWG can now also be used with the turbo scan. (#618)

### Changed
- makeDataSetxxx methods generic functionality split off. Added a warning for data shape differences (#598, #602). 
- Allow plotCallback to operate on datetime axis (#584).
- VirtualDAC now has an option to set gate_map and instruments on VirtualDAC (#583).
- All unittests are moved to a tests folder (#574).
- Legacy code now throws exceptions (#551).
- VideoMode tool was restructured to clean up the interface (##594)

### Deprecated
- loadExperimentData and saveExperimentData are now deprecated functions (#581).

### Removed
...

### Fixed
- Fixed virtual AWG marker on the same channel (#599).
- Fixed the readthedocs build (#590).
- Fixed issue with HDAWG8 sampling rate (#589).
- Fixed Travis warnings and tests instrument_drivers (#569).
- Fixed Travis build related to Qcodes version numbering (#604, #606).

### Security
...

## [1.1.2] - 2019-04-12

### Added
- Added normalization function to virtual gate matrix (#465).
- Improved simulated digitizer (#464).
- Improved documentation for virtual gates object (#456).

### Changed
- Moved QC toolkit references to QuPulse (#455).
- Updated Fermi fit example (#451).

### Removed
- Removed old reference hints to spin-projects (#458).

### Fixed
- Fixed the installation instructions (#546, #547).
- Fixed the parameter viewer (#449).


## [1.1.0] - 2018-09-08

### Added
- Added example notebook for awg_to_plunger (#424).
- Added QTT to readthedocs (#419, #409).
- Added Autofit to anti-crossing example notebook (#422).
- Added QTT to Travis continues integration (#409, #411, #413, #414).
- Added option to save additional metadata to scan functions (#402).
- Added Zurich Instruments UHFLI to measure segment and video mode (#396).
- Added new virtual AWG with similar functionality (#366).
- Added the parameter invert to the fast_Tune function for RF readout (#312).

### Changed
- Improved algorithm documentation (#436).
- Updated fit Ramsey scan example notebook (#442).
- Updated polynomial fit example notebook (#441).
- Updated Spin qubit measurement example notebook (#433).
- Updated simple measurement example notebook (#432).
- Changed Fermi-linear fitting function (#428).
- Updated PAT analysis notebook (#421).
- Updated extracting lever arm and charging energy example notebook (#427).
- Updated RTS example notebook (#425).
- Moved plotPoints to qgeometry Removed save_instrument_json (#418).
- Moved scans to the measurements folder (#397).

### Removed
- Removed save_instrument_json (#418).
- Removed parameter scaler. Use QCoDeS version instead (#398).
- Removed reports and create double dot jobs (#397).

### Fixed
- Fixed units in pinchoff data (#443).
- Fixed some units in datasets (#440).
- Make sure units are saved and loaded with GNUPlotFormatter (#434).
- Fixed problems with the scan2Dturbo function (#423).
- Fixed parameter in widget (#384).


[Unreleased]: https://github.com/QuTech-Delft/qtt/compare/1.1.2...HEAD
[1.1.2]: https://github.com/QuTech-Delft/qtt/compare/v1.1.0...1.1.2
[1.1.0]: https://github.com/QuTech-Delft/qtt/releases/v1.1.0
