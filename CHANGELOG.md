# Changelog
All notable changes to the Quantum Technology Toolbox will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Dev

### Changed
- Replaced implementation of polyintersection from Polygon3 to shapely (#744)

## \[1.2.2] - 14-5-2020
### Added
- Add example notebook for mongodb (#679)
- Add analysis results to notes (#712)
- Update docstrings (#718)
- Add numdifftools to requirements (#726)
- Add fitting of sine wave (#731)

### Changed
- Prevent tests from writing data to the user directory (#710)
- Implement functionality for print_matrix argument (#714)
- Bump bleach from 3.1.0 to 3.1.1 (#717)
- scanjob_t uses SweepFixedValues (including 'end') instead of slices (excluding 'end') (#722)
- Bump bleach from 3.1.1 to 3.1.2 (#727)
- Allow plotting arguments to be passed in drawCrosshair (#728)
- Use lmfit in fit_gauss_ramsey. The first point of the data is now also used for fitting (#729)
- Bump bleach from 3.1.2 to 3.1.4 (#730)
- Update to sine fitting (#734)
- Replaced chi_squared with reduced chi_squared in allxy fit (#736)

### Removed
- The ttrace code has been deprecated and is removed in this release (#709)

### Deprecated
- Moved the deprecated loadOneDotPinchvalues to legacy.py (#722)

### Fixed
- Prevent tests from writing data to the user directory (#710)
- Disable all output channels when restarting the videomode (#711)
- Fix bug in HDAWG8 driver where the gain was set to the range, whereas it should be gain = range/2 (#720)
- Fix bug in generation of step values for vector scans (#723)
- Packages 'tests' are not installed anymore. These packages gave problems with loading similar named modules in other packages (#724)
- Fix for deprecated old style qupulse Sequencer test (#733)
- Update qtt to qcodes 0.13.0 (#735)
- Improved calculation of covariances (#736)
- Fixed failing unit test depending on order of peak_local_max (#741)

## \[1.2.1] - 15-1-2020
### Added
- add representation method for virtual awg (#697)

### Changed
- Cleanup result selection of sensingdot_t (#694)
- Update dataset processing (#701)
- derive calibration exception from normal exception (#704)
- use bionic as build env (#705)

### Removed
- Removed the QCoDeS StandardParameter from QTT (#700).

### Deprecated
- Deprecated live.py, live_plotting.py and tools.py removed

### Fixed
- Fixed the import of M4i driver in function get_sampling_frequency (#700).
- fix for period argument of sensingdot_t (#696)
- Fix create_vectorscan for the new virtual awg  (#702)
- fix deprecation warnings (#703)

## \[1.2.0] - 2019-12-17

## \[1.1.3] - Unreleased

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
- Add method to fit AllXY experiment (#664)
- Added more example datasets (#670)
- Add method to improve double gaussian fit by initialization based on large-small gaussian (#680)
- Add option to compensate for re-arm time of M4i trigger (#607).
- Add methods for performing operations on datasets (#593).
- The new virtual AWG can now also be used with the turbo scan (#618).
- Add re-arm compensation (#614).
- Add restrict_boundaries method to the VirtualDAC (#631).
- Add option to set dataset record label through scanjob (#655).
- Add calibration exception (#585).
- Add dataclasses to setup.py (#628).
- Add fraction up and down as output of rts (#623).
- Add integration tests back to automated testing (#603).
- Add integration tests for dataset (#567).
- Add option to perform filtering (#645).
- Add pyqt5 to setup.py; remove pandas (#621).
- Add unit field to parameter viewer (#566).
- Add unittest on loading a json and hdf5 format dataset (#647).
- Add smoothing to derivative of signal in Coulomb peak fitting (#641).
- Added option to enlargelims to set x and y independently (#646).
- Measurement according to Elzerman readout (#562).
- Added UHFLI integration (#573).
- Added example for scan1D with UHFLI (#561).
- Added VirtualAwgInstrumentAdapter (#588).
- Add update model for markov chains (#592).
- Implement signal processor and signal processor interface (#556).
- Virtual gate scans with overlapping gates (#626).
- Fitting of Gaussian using lmfit (#686)
- Method to convert lmfit fitting result to dictionary (#686)

### Changed
- makeDataSetxxx methods generic functionality split off. Added a warning for data shape differences (#598, #602).
- Allow plotCallback to operate on datetime axis (#584).
- VirtualDAC now has an option to set gate_map and instruments on VirtualDAC (#583).
- All unit tests/integration tests are moved to a separated tests folder (#574, #600).
- Legacy code now throws exceptions (#551).
- Name of chirp template is passed to QuPulse template (#668)
- VideoMode tool was restructured to clean up the interface (#594).
- Updates requirements on QCoDeS version (#614).
- qtt.data.dataset_labels works for 1D scans now (#629).
- Removed plunger argument functionality from tunnelrates_RTS (#625).
- Improved initial fit of Gauss Ramsey (#643, #661, #678)
- Double Gaussian fitting is faster (using lmfit) and has better initial estimates (#648).
- Updated json serialization code to use qilib (#630).
- Autodetect notebooks for Travis testing (#627).
- Improved installation procedure (#613).
- Updated docs of PPT methods (#550).
- Updated documentation (#555).
- Updated to upstream QCoDeS: import from plot packages instead of main module (#649).
- Raise exception when diffImageSmooth is called with incorrect arguments (#608).
- Move mock imports to unittest.mock (#620).
- Enable all output channels for virtual awg in videomode (#644).
- Refactor parameterviewer; remove default min max values for parameters (#570).
- Refactor part of the RTS code (#577).
- Moved fit_gaussian to fitting module (#686)

### Deprecated
- loadExperimentData and saveExperimentData are now deprecated functions (#581).

### Fixed
- Fixed virtual AWG marker on the same channel (#599).
- Fixed the readthedocs build (#590).
- Fixed issue with HDAWG8 sampling rate (#589).
- Fixed Travis warnings and tests instrument_drivers (#569).
- Fixed Travis build related to Qcodes version numbering (#604, #606).
- Fixed incorrect initial estimate of fit_gaussian (#671)
- Fixed issue with setSingleStep (#633).
- QCodes 0.6.0 fix (#660).
- Fix matplotlib callback for NaN entries (#662).
- Fixed time axis for m4i acquire segments (#638).
- Fixed Travis build for VideoMode (#637).
- Resolved circular dependency problem in qtt and qilib (#609).
- Fix bug in awg_to_plunger method (#658).
- Fix default_parameter_name in scan method; fix issue with delete argument in scans1Dfast (#568).
- Fix definition of chirp signal (#622).
- Fix to acquire_segments (#575).
- Fixes to cc measurements (#572).
- Fix qilib for readthedocs build (#624).
- Fix coulomb example notebook (#619).
- VirtualDacInstrumentAdapter needs to set instead of append instruments (#605).
- Fixes for transition of m4i to qcodes contrib (#692).

## \[1.1.2] - 2019-04-12

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

## \[1.1.0] - 2018-09-08

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

[Unreleased]: https://github.com/QuTech-Delft/qtt/compare/1.2.1...HEAD
[1.2.1]: https://github.com/QuTech-Delft/qtt/compare/1.2.0...1.2.1
[1.2.0]: https://github.com/QuTech-Delft/qtt/compare/1.1.2...1.2.0
[1.1.2]: https://github.com/QuTech-Delft/qtt/compare/v1.1.0...1.1.2
[1.1.0]: https://github.com/QuTech-Delft/qtt/releases/v1.1.0
