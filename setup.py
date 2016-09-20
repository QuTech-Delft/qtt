from setuptools import setup, find_packages
from distutils.version import StrictVersion
from importlib import import_module


def readme():
    with open('README.md') as f:
        return f.read()

extras = {
    'Numpy': ('numpy', '1.9'),
    'MatPlot': ('matplotlib', '1.5'),
#    'QtPlot': ('pyqtgraph', '0.9.10'),
    'SciPi': ('scipy', '0.15'),
    'qcodes': ('qcodes', '0.1'),
    'scikit-image': ('skimage', '0.11'),
    'pandas': ('pandas', '0.15'),
    'Polygon3': ('Polygon', '3.0'),
    'hickle': ('hickle', '2.0'),
    'h5py': ('h5py', '0.1'),
}
extras_require = {k: '>='.join(v) for k, v in extras.items()}

setup(name='qtt',
      version='0.1.0',
      use_2to3=False,
      author='Pieter Eendebak',
      author_email='pieter.eendebak@tno.nl',
      maintainer='Pieter Eendebak',
      maintainer_email='pieter.eendebak@tno.nl',
      description='Python-based framework for analysis and tuning of quantum dots',
      long_description=readme(),
      url='http://qutech.nl',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
 	  'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Topic :: Scientific/Engineering'
      ],
      license='Private',
      # if we want to install without tests:
      # packages=find_packages(exclude=["*.tests", "tests"]),
      packages=find_packages(),
      requires=['numpy', 'matplotlib', 'scipy'], # 'pandas', 'pyqt', 

      install_requires=[
          'numpy>=1.10',
          'IPython>=0.1',
          # nose is only for tests, but we'd like to encourage people to run tests!
          #'nose>=1.3',
      ],
      extras_require=extras_require,
      )

version_template = '''
*****
***** package {0} must be at least version {1}.
***** Please upgrade it (pip install -U {0}) in order to use {2}
*****
'''

missing_template = '''
*****
***** package {} not found
***** Please install it in order to use {}
*****
'''

# now test the versions of extras
for extra, (module_name, min_version) in extras.items():
    try:
        module = import_module(module_name)
        if StrictVersion(module.__version__) < StrictVersion(min_version):
            print(version_template.format(module_name, min_version, extra))
    except ImportError:
        print(missing_template.format(module_name, extra))

