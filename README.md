# QuTech Tuning

QTT is a Python-based framework developed by QuTech (TU Delft/TNO) for automated tuning of qubits.

Also see
- [QuTech](https://http://qutech.nl/)
- [Qcodes](https://github.com/qdev-dk/Qcodes)
- [TNO](https://tno.nl)

QTT is compatible with Python 3.5+.

## Structure

QTT depends on Qcodes. Other repositories are the `user` scripts and `stations` (measurement setups).

* Qcodes: the basic framework
* (qtt)[https://github.com/VandersypenQutech/qtt]: contains additional functionality and algorithms
* (stations)[https://github.com/VandersypenQutech/stations]: Contains hardware descriptions for experimential setups. There should be only one version of this repository and it should always match the current hardware.
* (users)[https://github.com/VandersypenQutech/users]: contains scripts and functions written by individual users

## Installation

For more detailed instructions read the file [INSTALL.md](INSTALL.md).

Make sure you have a working Python distribution. We recommend [Anaconda](https://www.continuum.io/downloads) as an easy way to get most of the dependencies out-of-the-box.

As the project is still private, install it directly from this repository:

- Clone this repositories somewhere on your hard drive. If you're using command line git, open a terminal window in the directory where you'd like to put qcodes and type:
```
> cd [MYLOCALDIR]
> git clone https://github.com/VandersypenQutech/Qcodes.git
> git clone https://github.com/VandersypenQutech/qtt.git
> git clone https://github.com/VandersypenQutech/users.git
> git clone https://github.com/VandersypenQutech/stations.git
```

- Install necessary python dependencies and install the python packages.

### Updating QTT

If you registered Qcodes with Python via `setup.py develop`, all you need to do to get the latest code is open a terminal window pointing to anywhere inside the repository and run `git pull`

## Usage

See the [docs](docs) directory (to be constructed)

For a general introduction also see
* [Introduction to Github](https://guides.github.com/activities/hello-world/)
* [Scientific python lectures](https://github.com/jrjohansson/scientific-python-lectures)

* If you use [Spyder](https://github.com/spyder-ide/spyder) then use the following settings:
- Use a `IPython` console and set the IPhyton backend graphics option to `QT`. This ensures correctly displaying the `Paramater viewer` and `DataBrowser`
- In Tools->Preferences->Console->Advanced settings uncheck the box `Enable UMR`

## Contributing

See [Contributing](CONTRIBUTING.md) for information about bug/issue reports, contributing code, style, and testing


## License

See [License](LICENSE.txt)
