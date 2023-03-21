[![Documentation Status](https://readthedocs.org/projects/qtt/badge/?version=latest)](https://qtt.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/qtt.svg)](https://badge.fury.io/py/qtt)

# Welcome

Welcome to the QTT framework. This README will shortly introduce the framework, and it will guide you through the structure, installation process and how to contribute. We look forward to working with you!

# Quantum Technology Toolbox

Quantum Technology Toolbox (QTT) is a Python-based framework developed initially by QuTech for the tuning and calibration of
quantum dots and spin qubits. [QuTech](http://qutech.nl) is an advanced research center based in Delft, the Netherlands, for quantum
computing and quantum internet, a collaboration founded by the [University of Technology Delft](https://www.tudelft.nl/en) (TU Delft) and
the Netherlands Organisation for Applied Scientific Research ([TNO](https://www.tno.nl/en)).

For usage of QTT see the detailed [documentation](https://qtt.readthedocs.io/en/latest/) on readthedocs.io.

QTT is the framework on which you can base your measurement and analysis scripts. QTT is based
on [QCoDeS](https://github.com/qdev-dk/Qcodes) (basic framework such as instrument drivers, DataSet) and the [SciPy](https://www.scipy.org/) ecosystem.

## Installation

QTT is compatible with Python 3.8+. QTT can be installed as a pip package to be used in a (virtual) Python environment.
We assume that software packages like [git](https://git-scm.com/downloads) and [python](https://www.python.org/downloads/)
are already installed on your system.

Note: when running Ubuntu Linux, installing these packages is done via:

```bash
sudo apt install git gcc python3.11 python3.11-venv python3.11-dev
```

for Python 3.11.x. Other Linux distributions require similar steps.

### Setting up a virtual environment

To create a clean virtual Python environment for your QTT development do:

```bash
mkdir qtt
cd qtt
```

Now activate the virtual environment. On Linux do:

```bash
python3 -m venv env
. ./env/bin/activate
```

or

```bash
source ./env/bin/activate
```

On Windows do:

```bash
python -m pip install virtualenv
python -m virtualenv --copies env
env\Scripts\activate.bat
```

Now we are ready to install QTT.

### Installation from PyPI

To use QTT, install it as a pip package:

```bash
pip install qtt
```

or install QTT from source.

### Installing from source

The source for QTT can be found at Github.
For the default installation from the QTT source directory execute:

```bash
git clone https://github.com/QuTech-Delft/qtt.git
cd qtt
pip install wheel
```

For QTT development install QTT in editable mode:

```bash
pip install -e .
```

For non-editable mode do:

```bash
pip install .
```

When (encountered on Linux) PyQt5 gives an error when installing try upgrading pip

```bash
pip install --upgrade pip
```

and rerun the respective install command.

### When incompatibility problems arise
**Note: This step is not meant for python >=3.11**
Sometimes the default installation does not work because of incompatible dependencies between the used packages
on your system. To be sure you use all the right versions of the packages used by QTT and its dependencies do:

```bash
pip install . -r requirements_lock_py310.txt
```

or for development

```bash
pip install -e . -r requirements_lock_py310.txt
```

This will install a tested set of all the packages QTT depends on.

### Testing

Tests for the QTT packages are contained in the subdirectory `tests`. To run the tests run the following command:

```bash
pytest
```

When integration tests fail because of errors in plotting try downgrading opencv-python to 4.2.0.34:

```bash
pip install opencv-python==4.2.0.34
```

When running on Windows Sysbsystem for Linux (WSL) you may need to uninstall opencv and install the headless version:

```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

### Installing for generating documentation

To install the necessary packages to perform documentation activities for QTT do:

```bash
pip install -e .[rtd]
```

The documentation generation process is dependent on pandoc. When you want to generate the
documentation and pandoc is not yet installed on your system navigate
to [Pandoc](https://pandoc.org/installing.html) and follow the instructions found there to install pandoc.
To build the 'readthedocs' documentation do:

```bash
cd docs
make html
```

### Vandersypen research group

For the Vandersypen research group there are more detailed instructions, read the file INSTALL.md in the spin-projects
repository.

### Updating QTT

To update QTT do:

```bash
pip install . --upgrade
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
