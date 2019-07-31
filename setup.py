import platform
import re
from importlib import import_module

from setuptools import find_packages, setup
from setuptools._vendor.packaging.version import Version, InvalidVersion


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


extras = {
    # name: (module_name, minversion, pip_name)
    'scikit-image': ('skimage', '0.11', 'scikit-image'),
    'pandas': ('pandas', '0.15', None),
    'attrs': ('attr', '16.2.0', 'attrs'),
    'h5py': ('h5py', '0.1', None),
    'pyzmqrpc': ('zmqrpc', '1.5', None),
    'pytables': ('tables', '3.2', None),
    'apscheduler': ('apscheduler', '3.4', None),
    'Polygon3': ('Polygon', '0.1', None),
    'pyqt5': ('PyQt5', '0.11', 'pyqt5'),
}

if platform.system() == 'Windows':
    extras['pywin32'] = ('win32', '0.1', None)

extras_require = {k: '>='.join(v[0:2]) for k, v in extras.items()}

print('packages: %s' % find_packages())

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
      packages=find_packages(where='./src'),
      install_requires=[
          'matplotlib>=3.0', 'pandas', 'attrs', 'dulwich', 'qtpy', 'nose', 'hickle', 'pyzmqrpc',
          'numpy>=1.15', 'scikit-image', 'IPython>=0.1', 'qcodes>=0.4', 'Polygon3',
          'scipy', 'pyqtgraph', 'qupulse'
      ],
      tests_require=['unittest'],
      extras_require=extras_require,
      zip_safe=False,
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

invalid_version_template = '''
*****
***** package {0} has an invalid version: {1}
*****
'''

# now test the versions of extras
for extra, (module_name, min_version, pip_name) in extras.items():
    try:
        module = import_module(module_name)
    except ImportError:
        print(missing_template.format(module_name, extra))
        continue

    try:
        fnd_version = Version(module.__version__)
    except AttributeError:
        # probably a package not providing the __version__ attribute
        pass
    except InvalidVersion:
        print(invalid_version_template.format(module_name, module.__version__))
    else:
        if fnd_version < Version(min_version):
            print(version_template.format(module_name, min_version, extra))
