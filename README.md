# Welcome

Welcome to the QTT framework. This README will shortly introduce the framework, and it will guide you through the structure, installation process and how to contribute. We look forward to working with you!

# Quantum Technology Toolbox

Quantum Technology Toolbox (QTT) is a Python-based framework developed initially by QuTech for the tuning and calibration of
quantum dots and spin qubits. [QuTech](http://qutech.nl) is an advanced research center based in Delft, the Netherlands, for quantum
computing and quantum internet, a collaboration founded by the [University of Technology Delft](https://www.tudelft.nl/en) (TU Delft) and
the Netherlands Organisation for Applied Scientific Research ([TNO](https://www.tno.nl/en)).

For usage of QTT see the detailed [documentation](https://qtt.readthedocs.io/en/latest/) on readthedocs.io.


QTT is the framework on which you can base your measurement and analysis scripts. QTT is based
on [Qcodes](https://github.com/qdev-dk/Qcodes) (basic framework such as instrument drivers, DataSet) and the [SciPy](https://www.scipy.org/) ecosystem.

## Installation

QTT is compatible with Python 3.5+. QTT can be installed as a pip package to be used in a (virtual) Python environment.
We assume that software packages like [git](https://git-scm.com/downloads) and [python](https://www.python.org/downloads/)
are already installed on your system.

Note: when running Ubuntu Linux, installing these packages is done via:
```
$ sudo apt install git gcc python3 python3-venv python3-dev
```
for Python 3.6.x or
```
$ sudo apt install git gcc python3.7 python3.7-venv python3.7-dev
```
for Python 3.7.x. Other Linux distributions require similar steps.

### Setting up a virtual environment
To create a clean virtual Python environment for your QTT development do:
```
$ mkdir qtt
$ cd qtt
```
Now activate the virtual environment. On Linux do:
```
$ python3 -m venv env
$ . ./env/bin/activate
 or
$ source ./env/bin/activate
```
or use the python3.7 equivalent to generate a python 3.7 environment.
On Windows do:
```
$ python -m venv env
$ env\Scripts\activate.bat
```
Now we are ready to install QTT.
### Installation from Pypi
To use QTT, install it as a pip package: 
```
$ pip install qtt
```
### Installing from source
For QTT development we advice to install from source. The source for QTT can be found at Github.
For the default installation from the QTT source directory execute:
```
$ git clone https://github.com/QuTech-Delft/qtt.git
$ cd qtt
$ pip install wheel
$ pip install -e .
```
### When incompatibility problems arise
Sometimes the default installation does not work because of incompatible dependencies between the used packages
on your system. To be sure you use all the right versions of the packages used by QTT and its dependencies do:
```
$ pip install . -r requirements_lock.txt
```
or for development
```
$ pip install -e . -r requirements_lock.txt
```
This will install a tested set of all the packages QTT depends on.

### Testing

Tests for the QTT packages are contained in the subdirectory `tests`. To run the tests run the following command:

```
$ pytest
```
### Installing for generating documentation
To install the necessary packages to perform documentation activities for QTT do:
```
$ pip install -e .[rtd]
```
The documentation generation process is dependent on pandoc. When you want to generate the
documentation and pandoc is not yet installed on your system navigate
to [Pandoc](https://pandoc.org/installing.html) and follow the instructions found there to install pandoc. 
To build the 'readthedocs' documentation do:
```
$ cd docs
$ make html
```
### Vandersypen research group

For the Vandersypen research group there are more detailed instructions, read the file INSTALL.md in the spin-projects
repository.

### Updating QTT

To update QTT do:
```
$ pip install . --upgrade
```

## Usage

See the [documentation](https://qtt.readthedocs.io/en/latest/) and the example notebooks in the [docs/notebooks](docs/notebooks) directory.

For a general introduction also see
* [Introduction to Github](https://guides.github.com/activities/hello-world/)
* [Scientific python lectures](https://github.com/jrjohansson/scientific-python-lectures)

If you use [Spyder](https://github.com/spyder-ide/spyder) then use the following settings:
  - Use a `IPython` console and in `Tools->Preferences->IPython console->Graphics` set the IPython backend graphics option to `Qt5`. This ensures correctly displaying the `ParameterViewer` and `DataBrowser`
  - In `Tools->Preferences->Console->Advanced settings` uncheck the box `Enable UMR`

## Contributing

See [Contributing](CONTRIBUTING.md) for information about bug/issue reports, contributing code, style, and testing

## License

See [License](LICENSE.txt)
