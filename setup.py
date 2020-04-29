import os
from inspect import getsourcefile
from os.path import abspath
import platform
import re

from setuptools import find_packages, setup


def readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()


def get_version(verbose=1, filename='src/qtt/version.py'):
    """ Extract version information from source code """

    with open(filename, 'r') as f:
        ln = f.readline()
        m = re.search('.* ''(.*)''', ln)
        version = (m.group(1)).strip('\'')
    if verbose:
        print('get_version: %s' % version)
    return version


tests_require = ['coverage', 'jupyter', 'mypy', 'pytest']

install_requires = [
    'apscheduler', 'attrs', 'dulwich', 'h5py', 'hickle', 'IPython>=0.1', 'lmfit', 'matplotlib>=3.0',
    'numpy>=1.15', 'opencv-python', 'PyQt5', 'pyqtgraph', 'pyvisa', 'pyzmqrpc', 'qcodes>=0.8.0',
    'qcodes-contrib-drivers', 'qilib', 'qtpy', 'qupulse', 'redis', 'scipy>=0.18', 'scikit-image', 'jupyter',
    'coverage', 'sympy', 'numdifftools'
] + tests_require

if platform.system() == 'Windows':
    install_requires.append('pywin32')

if platform.python_version() < "3.7.0":
    install_requires.append('Polygon3')
else:
    if platform.system() == 'Windows':
        # When Polygon3 not yet in git-repository get it locally
        file_path = abspath(getsourcefile(lambda: 0))
        bin_dir = os.path.join(os.path.dirname(file_path), 'bin')
        Polygon3_local = f'Polygon3 @ file://{bin_dir}/Polygon3-3.0.8-cp37-cp37m-win_amd64.whl'
        Polygon3_git = 'Polygon3 @ https://github.com/QuTech-Delft/qtt/bin/Polygon3-3.0.8-cp37-cp37m-win_amd64.whl'
        Polygon3 = Polygon3_local
    else:
        Polygon3 = 'Polygon3'
    install_requires.append(Polygon3)

rtd_requires = [
    'sphinx>=1.7', 'sphinx_rtd_theme', 'nbsphinx', 'sphinx-automodapi'
]

extras_require = {"rtd": rtd_requires}

setup(name='qtt',
      version=get_version(),
      use_2to3=False,
      author='Pieter Eendebak',
      author_email='pieter.eendebak@tno.nl',
      maintainer='Pieter Eendebak',
      maintainer_email='pieter.eendebak@tno.nl',
      description='Python-based framework for analysis and tuning of quantum dots',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='http://qutech.nl',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering'
      ],
      license='MIT',
      package_dir={'': 'src'},
      packages=find_packages(where='./src', exclude=["*tests*"]),
      data_files=[('bin',
                  ['bin/Polygon3-3.0.8-cp37-cp37m-win_amd64.whl'])],
      install_requires=install_requires,
      tests_require=tests_require,
      extras_require=extras_require,
      zip_safe=False,
      )
