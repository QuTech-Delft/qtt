# Installation

## Install GIT

For the code we use git as the version control system. For Unix systems (Ubuntu) git can be installed with

> sudo apt-get install git

For windows we recommend the command line tools from https://git-scm.com/download/win

If you want a GUI you can use either
* https://tortoisegit.org/
* https://desktop.github.com/


## Create a working Python environment

You need 3.4+, but we recommand Python 3.5+. For Unix python is installed by default, for windows 
we recommend using [Anaconda](https://www.continuum.io/downloads). For Windows you need admin rights, or create
 your [own environment](http://conda.pydata.org/docs/using/envs.html).


## Install necessary packages
```
> conda create -n [yourname] python=3.5
> activate [yourname]
> conda install -y numpy matplotlib scipy spyder jupyter pyqt h5py attrs pandas
> pip install pyqtgraph pyvisa
> # old command: conda install -c conda-forge pyqtgraph>=0.10
> conda install -y coverage nose scikit-image qtpy graphviz pytest pywin32
> conda install -c https://conda.binstar.org/menpo opencv3
```
There is a bug in qtconsole, see [qtconsole#145](https://github.com/jupyter/qtconsole/pull/145), so do
```
> pip install git+https://github.com/jupyter/qtconsole.git
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

## Clone the necessary GIT repositories

All code is hosted on GitHub. If you are unfamiliar with git, read an [introduction](https://guides.github.com/activities/hello-world/) first.

There are four repositories:

1. QCodes
2. QTT
3. Users
4. Stations

The git commands are:
```
> cd [MYLOCALDIR]
> git clone https://github.com/VandersypenQutech/Qcodes.git
> git clone https://github.com/VandersypenQutech/qtt.git
> git clone https://github.com/VandersypenQutech/users.git
> git clone https://github.com/VandersypenQutech/stations.git
```

(optional) Go to the Qcodes directory and checkout the latest spin-qubit devel branch `sq10`
```
> cd [MYLOCALDIR]/Qcodes
> git checkout sq10
```

## Install packages

- Register the repositories with Python. For each of the repositories run this from the root directory:
```
> conda develop .  (for Anaconda)
> python setup.py develop --user  (for other systems)
```

Note: the following does NOT work with anaconda
 > python setup.py develop --user


## Hardware 

- If necessary install the drivers for your hardware. Some links:
* (Virtual COM port driver)[http://www.ftdichip.com/Drivers/VCP.htm]
* (GPIB USB interface)[http://www.ni.com/download/ni-488.2-16.0.0/6132/en/]


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

spyder --show-console --new-instance -p d:\users\%username% -w d:\users\%USERNAME% --window-title %USERNAME%

#echo "Press ..."
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

## Install redis

For Windows install Chocolately and then do
```
> choco install redis-64
> pip install redis
```
To start the redis server use `redis-server.exe` from the command line.
There are also methods/packages to install the redis server as a service, see
??? .

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


