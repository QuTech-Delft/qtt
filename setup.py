from setuptools import setup, find_packages
from distutils.version import StrictVersion
from importlib import import_module
import platform
import re

def readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()

def get_version(verbose=1, filename='qtt/version.py'):
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
    'Numpy': ('numpy', '1.9', None),
    'MatPlot': ('matplotlib', '1.5', None),
    'SciPi': ('scipy', '0.19', None),
    'qcodes': ('qcodes', '0.1.5', None),
    'scikit-image': ('skimage', '0.11', 'scikit-image'),
    'pandas': ('pandas', '0.15', None),
    'attrs': ('attr', '16.2.0', 'attrs'),
    'h5py': ('h5py', '0.1', None),
    'slacker': ('slacker', '0.1', None),
    'pyzmqrpc': ('zmqrpc', '1.5', None),
    'pytables': ('tables', '3.2', None),    
    'colorama': ('colorama', '0.1', None),    
    'apscheduler': ('apscheduler', '3.4', None),    
    'Polygon3': ('Polygon', '0.1', None),    
    'pyqtgraph': ('pyqtgraph', '0.11', None),    
    'pyqt5': ('PyQt5', '0.11', 'pyqt5'),    
}

if platform.system()=='Windows':
    extras['pywin32'] =  ('win32', '0.1', None)

extras_require = {k: '>='.join(v[0:2]) for k, v in extras.items()}

print('packages: %s' % find_packages())

try:
	import qcodes
except ImportError as ex:
	raise Exception('please install qcodes before running setup.py')
	
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
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering'
      ],
      license='Private',
      # if we want to install without tests:
      # packages=find_packages(exclude=["*.tests", "tests"]),
      packages=find_packages(),
      #requires=['numpy', 'matplotlib', 'scipy', 'qcodes', 'pandas', 'attrs', 'qtpy', 'slacker', 'nose', 'hickle'],
      install_requires=[
          'matplotlib', 'pandas', 'attrs', 'dulwich', 'qtpy', 'nose', 'slacker', 'hickle', 'pyzmqrpc',
          'numpy>=1.10', 'scikit-image', 'lmfit',
          'IPython>=0.1',
          'qcodes>=0.1.5',
          'Polygon3',
          'scipy'
          # nose is only for tests, but we'd like to encourage people to run tests!
          #'nose>=1.3',
      ],
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

# now test the versions of extras
for extra, (module_name, min_version, pip_name) in extras.items():
    try:
        module = import_module(module_name)
        if StrictVersion(module.__version__) < StrictVersion(min_version):
            print(version_template.format(module_name, min_version, extra))
    except AttributeError:
        # probably a package not providing the __version__ attribute
        pass
    except ImportError:
        print(missing_template.format(module_name, extra))
