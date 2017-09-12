# Installation

## Install GIT

For the code we use git as the version control system. For Unix systems (Ubuntu) git can be installed with

> sudo apt-get install git

For windows we recommend the command line tools from https://git-scm.com/download/win

If you want a GUI you can use either
* https://tortoisegit.org/
* https://desktop.github.com/

## Clone the necessary GIT repositories

All code is hosted on GitHub. If you are unfamiliar with git, read an [introduction](https://guides.github.com/activities/hello-world/) first.

There are four repositories:

1. QCodes
2. QTT
3. Spin-projects

The git commands are:
```
> cd [MYLOCALDIR]
> git clone https://github.com/VandersypenQutech/Qcodes.git
> git clone https://github.com/VandersypenQutech/qtt.git
> git clone https://github.com/VandersypenQutech/spin-projects.git
```

Optional:
```
> git clone https://github.com/DiCarloLab-Delft/PycQED_py3.git
```

## Create a working Python environment

You need 3.4, but we recommand Python 3.6+. For Unix python is installed by default, for windows 
we recommend using [Anaconda](https://www.continuum.io/downloads). For Windows you need admin rights, or create
 your [own environment](http://conda.pydata.org/docs/using/envs.html).


## Install necessary packages

Go to the location `[MYLOCALDIR]`/qtt and run
```
> conda env create -n [yourname] -f condalist.yml
> activate [yourname]
```

For Mac OS with anaconda type from the command line:
```
> cd [MYLOCALDIR]/qtt
> conda install --file requirements_mac.txt
> conda install -c menpo opencv3
> conda install -c nmearl pyqtgraph
> pip install pyvisa
```
(For Mac OS using Python 3.4, follow instruction in this [blog post](http://www.pyimagesearch.com/2015/06/29/install-opencv-3-0-and-python-3-4-on-osx/) to install `openCV`)



## Install packages

- Register the qcodes, qtt and spin-projects repositories with Python. Open a command prompt, change to your environment and run the following command
in the directories `Qcodes`, `qtt` and `spin-projects/stations`.
```
> python setup.py develop    (Anaconda, while in your environment)
```

Note: the following does NOT work with anaconda
```
> python setup.py develop --user
```
For Unix systems the proper command is `python setup.py develop --user`.

## Hardware 

- If necessary install the drivers for your hardware. Some links:
* [Virtual COM port driver](http://www.ftdichip.com/Drivers/VCP.htm) (used for the IVVI-rack)
* [GPIB USB interface](http://www.ni.com/download/ni-488.2-16.0.0/6132/en/)
* [Spectrum M4i](http://spectrum-instrumentation.com/en/m4i-platform-overview)
* [SPI-rack](https://github.com/Rubenknex/SPI-rack)

## Spyder

* Use a IPython console and set the IPhyton backend graphics option to QT5. This ensures
 correctly displaying the Paramater viewer and DataBrowser
* In Tools->Preferences->Python interpreter, uncheck the box Enable UMR 

## Create startup shortcuts

For Spyder one can use something like:

```
@echo off

set USERNAME=eendebakpt
set QTTUSERDIR=D:\users\%USERNAME%\users\%USERNAME%
set QCODESFRONTEND=spyder

echo "Starting spyder" 
SET PYTHONPATH=%PYTHONPATH%;%QTTUSERDIR%
call activate %USERNAME%
d:
cd d:\users\%USERNAME%

rem set SPYDER_DEBUG=1
spyder --show-console --new-instance -p d:\users\%username% -w d:\users\%USERNAME% --window-title %USERNAME%
rem --multithread 

rem echo "Press ..."
pause

call deactivate %USERNAME%
```

For a notebook session:

```
@echo off
set USERNAME=eendebakpt

echo "Starting Jupyter notebook" 
call activate %USERNAME%
d:
cd d:\users\%USERNAME%
jupyter notebook
call deactivate %USERNAME%
```

## Hickle (optional)

To install dev version of hicke:
```
pip install git+https://github.com/telegraphic/hickle.git@dev
```

## Install redis

For Windows install redis from https://github.com/MSOpenTech/redis.
To start the redis server use `redis-server.exe` from the command line (if this is not done automatically).

For Unix systems follow the instructions of your OS. For Ubuntu it is:
```
sudo apt-get install redis-server
sudo apt-get install python3-redis
```


## Git credentials

```
git config credential.username [peendebak]
git config credential.useHttpPath true
git config credential.helper cache
git config credential.helper store	# stores passwords in plain text!
```

## Python warnings

```
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
```

### Old commands

Manual installation:
```
> conda install numpy scipy pyqtgraph spyder nose pandas scikit-learn scikit-image rope jupyter matplotlib h5py
> pip install slacker attrs pyserial redis pyvisa
> conda install -c conda-forge opencv
```

There was a bug in qtconsole, see [qtconsole#145](https://github.com/jupyter/qtconsole/pull/145), so do
```
> pip install git+https://github.com/jupyter/qtconsole.git
```
Custom pyqtgraph:
```
> # old command: conda install -c conda-forge pyqtgraph>=0.10
```

The menpo opencv package is not yet available for python 3.6. For older versions:
```
> conda install -c https://conda.binstar.org/menpo opencv3
```
```
> conda create -n [yourname] python=3.6 numpy matplotlib scipy spyder jupyter pyqt h5py pandas pyqtgraph
> activate [yourname]
> pip install pyvisa attrs pyserial redis
> conda install -y coverage nose scikit-image qtpy graphviz pytest pywin32
> # opencv from http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv
> pip install K:\ns\qt\spin-qubits\software\qtt\opencv_python-3.2.0+contrib-cp36-cp36m-win_amd64.whl
> python -c "import wget;"
```

Redis from [chocolately](https://chocolatey.org/)
```
> choco install redis-64
> pip install redis
```

Opencv pre-build packages:
```
> # opencv from http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv
> pip install K:\ns\qt\spin-qubits\software\qtt\opencv_python-3.2.0+contrib-cp36-cp36m-win_amd64.whl
```