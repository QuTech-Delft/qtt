# Changelog
All notable changes to the Quantum Tuning Toolkit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]
# TODO...

### Added
### Changed
### Deprecated
### Removed
### Fixed
### Security


## [1.1.2] - 2018-04-12

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
