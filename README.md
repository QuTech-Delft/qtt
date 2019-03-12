# Welcome

Welcome to the QTT framework. This README will shortly introduce the framework, and it will guide you through the structure, installation process and how to contribute. We look forward to working with you!

# Quantum Technology Toolbox

Quantum Technology Toolbox (QTT) is a Python-based framework developed initially by QuTech for the tuning and calibration of
quantum dots and spin qubits. [QuTech](http://qutech.nl) is an advanced research center based in Delft, the Netherlands, for quantum
computing and quantum internet, a collaboration founded by the [University of Technology Delft](https://www.tudelft.nl/en) (TU Delft) and
the Netherlands Organisation for Applied Scientiﬁc Research ([TNO](https://www.tno.nl/en)).

For usage of QTT see the detailed [documentation](https://qtt.readthedocs.io/en/latest/) on readthedocs.io.


QTT is the framework on which you can base your measurement and analysis scripts. QTT is based
on [Qcodes](https://github.com/qdev-dk/Qcodes) (basic framework such as instrument drivers, DataSet) and the [SciPy](https://www.scipy.org/) ecosystem.

## Installation

QTT is compatible with Python 3.5+. QTT can be installed as a pip package:
```
$ pip install qtt
```
For development we advice to install from source. First retrieve the source code using git, and then install from the qtt source directory using the command:
```
$ python setup.py develop
```

For the Vandersypen research group there are more detailed instructions, read the file INSTALL.md in the spin-projects repository.

### Updating QTT

If you registered qtt with Python via `setup.py develop`, all you need to do to get the latest code is open a terminal window pointing to anywhere inside the repository and run `git pull`

## Usage

See the [documentation](https://qtt.readthedocs.io/en/latest/) and the example notebooks in the [docs/notebooks](docs/notebooks/) directory.

For a general introduction also see
  * [Introduction to Github](https://guides.github.com/activities/hello-world/)
  * [Scientific python lectures](https://github.com/jrjohansson/scientific-python-lectures)

If you use [Spyder](https://github.com/spyder-ide/spyder) then use the following settings:
   - Use a `IPython` console and in `Tools->Preferences->IPython console->Graphics` set the IPython backend graphics option to `Qt5`. This ensures correctly displaying the `ParameterViewer` and `DataBrowser`
   - In `Tools->Preferences->Console->Advanced settings` uncheck the box `Enable UMR`

## Contributing

See [Contributing](CONTRIBUTING.md) for information about bug/issue reports, contributing code, style, and testing

## Testing

Tests for the qtt packages are contained in the subdirectory `tests` and as test
functions (`test_*`) in the code. To run the tests you can run the following command:
```
$ pytest
```

## License

See [License](LICENSE.txt)
