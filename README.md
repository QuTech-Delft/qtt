# QuTech Tuning

QTT is a Python-based framework developed by QuTech (TU Delft/TNO) for tuning and calibration of spin-qubits.

Also see
- [QuTech](https://http://qutech.nl/)
- [Qcodes](https://github.com/qdev-dk/Qcodes)
- [TNO](https://tno.nl)

QTT is compatible with Python 3.5+.

## Structure

QTT depends on Qcodes. Other repositories are the `user` scripts and `stations` (measurement setups).

* Qcodes: the basic framework
* [qtt](https://github.com/VandersypenQutech/qtt): contains additional functionality and algorithms
* [spin-projects](https://github.com/VandersypenQutech/spin-projects): Contains hardware descriptions for experimential setups and project data.
  
## Installation

For more detailed instructions read the file [INSTALL.md](INSTALL.md).

### Updating QTT

If you registered Qcodes with Python via `setup.py develop`, all you need to do to get the latest code is open a terminal window pointing to anywhere inside the repository and run `git pull`

## Usage

See the [docs](docs) directory (to be constructed)

For a general introduction also see
* [Introduction to Github](https://guides.github.com/activities/hello-world/)
* [Scientific python lectures](https://github.com/jrjohansson/scientific-python-lectures)

* If you use [Spyder](https://github.com/spyder-ide/spyder) then use the following settings:
- Use a `IPython` console and set the IPyhton backend graphics option to `QT`. This ensures correctly displaying the `ParameterViewer` and `DataBrowser`
- In Tools->Preferences->Console->Advanced settings uncheck the box `Enable UMR`

## Contributing

See [Contributing](CONTRIBUTING.md) for information about bug/issue reports, contributing code, style, and testing

## Testing

Tests for the qtt packages are contained in the subdirectory `tests` and as test functions (`test_*`) in the code. To run the tests you can run one of the commands below. Note that only `pytest` runs also the tests throughout the code while `python qtt/test.py` only runs the ones in the tests directory.
```
> python qtt/test.py
> pytest # (Windows)
> py.test # (Unix)
```

## License

See [License](LICENSE.txt)
