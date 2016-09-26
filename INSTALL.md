# Installation

## Install GIT

For the code we use git as the version control system. For Unix systems (Ubuntu) git can be installed with

> sudo apt-get install git

For windows we recommend the command line tools from https://git-scm.com/download/win

If you want a GUI you can use either
* https://tortoisegit.org/
* https://desktop.github.com/


## Create a working Python environment

You need 3.4+, but we recommand Python 3.5. For Unix python is installed by default, for windows we recommend using Anaconda.
The installer can be found here

## Install necessary packages

> conda install numpy matplotlib scipy spyder jupyter pyqt
> conda install -c https://conda.binstar.org/nmearl pyqtgraph
> conda install coverage nose scikit-image qtpy graphviz
> pip install pyvisa
> conda install -c https://conda.binstar.org/menpo opencv3


## Clone the necessary GIT repositories

1) QCodes
2) QTT
3) VandersypenUsers
4) VandersypenSetups

The git links are:

https://github.com/qdev-dk/Qcodes.git
https://github.com/VandersypenQutech/qtt.git
https://github.com/VandersypenQutech/users.git
https://github.com/VandersypenQutech/stations.git

```
> cd [MYLOCALDIR]
> git clone https://github.com/qdev-dk/Qcodes.git
> git clone https://github.com/VandersypenQutech/qtt.git
> git clone https://github.com/VandersypenQutech/users.git
> git clone https://github.com/VandersypenQutech/stations.git
```

## Install packages

- Register the repositories it with Python. For each of the repositories run this from the root directory:
```
> conda develop ./  (for Anaconda)
> python setup.py develop --user  (for other systems)
```

Note: the following does NOT work with anaconda
 > python setup.py develop --user

## Create startup shortcuts

For Spyder one can use something like:

```
@echo off
echo "Starting spyder" 
call activate eendebakpt
d:
cd d:\users\eendebakpt
spyder --new-instance -w d:\users\eendebakpt 
#spyder --show-console --new-instance -w d:\users\eendebakpt --session=eendebakpt.session.tar 
call deactivate eendebakpt
```

For a notebook session:

```
@echo off
echo "Starting Jupyter notebook" 

call activate eendebakpt
d:
cd d:\users\eendebakpt
jupyter notebook
call deactivate eendebakpt
```

## Install redis

For Windows install Chocolately and then do
```
> choco install redis064
> pip install redis
```
For Unix systems follow the instructions of your OS.

## Git credentials

```
git config credential.username [peendebak]
git config credential.useHttpPath true
git config credential.helper cache
git config credential.helper store	# stores passwords in plain text!
```


