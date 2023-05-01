""" Quantum Technology Toolbox

The QTT package contains functionality for the tuning and calibration of spin-qubits. The package is
divided into subpackages:

    - Measurements: functionality to perform measurements on devices
    - Algorithms: functionality to analyse measurements
    - Simulation: contains simulations of quantum dot systems
    - Tools: misc tools
    - Gui: Several gui element for visualization of data
    - Instrument drivers: contains QCoDeS drivers for various instruments

For more information see https://github.com/qutech-delft/qtt

"""
# flake8: noqa (we don't need the "<...> imported but unused" error)

import copy
import importlib
import warnings

import qcodes
import qcodes_loop
from qcodes import Instrument, ManualParameter, Parameter, Station
from qcodes_loop.data.location import FormatLocation
from setuptools._vendor.packaging.version import Version

import qtt.algorithms
import qtt.data
import qtt.exceptions
import qtt.measurements
import qtt.utilities.tools
from qtt.gui.live_plotting import start_measurement_control
from qtt.version import __version__

try:
    import pyqtgraph

    import qtt.gui.live_plotting
    import qtt.gui.parameterviewer
    from qtt.gui.dataviewer import DataViewer
    from qtt.gui.parameterviewer import createParameterWidget
except ImportError:
    # no gui available
    warnings.warn('pyqtgraph could not be imported, gui elements not available')


# %% Check packages


def check_version(version, module=qcodes, optional=False, install_message=None):
    """ Check whether a module has the correct version """
    if isinstance(module, str):
        try:
            m = importlib.import_module(module)
            module = m
        except ModuleNotFoundError:
            if optional:
                warnings.warn('optional package %s is not available' %
                              module, qtt.exceptions.MissingOptionalPackageWarning)
                return
            else:
                if install_message is not None:
                    print(install_message)
                raise Exception('could not load module %s' % module)

    mversion = getattr(module, '__version__', None)
    if mversion is None:
        raise Exception(' module %s has no __version__ attribute' % (module,))

    if Version(mversion) < Version(version):
        if optional:
            warnings.warn('package %s has incorrect version' % module, qtt.exceptions.PackageVersionWarning)
        else:
            raise Exception(' from %s need version %s (version is %s)' % (module, version, mversion))


# we make an explicit check on versions, since people often upgrade their
# installation without upgrading the required packages
check_version('1.0', 'qtpy')
check_version('0.18', 'scipy')
check_version('0.1', 'redis', optional=True)
check_version('0.23.0', qcodes)
check_version('0.2', 'qupulse')


# %% Add hook to abort measurement

# connect to redis server
_redis_connection = None
try:
    import redis
    _redis_connection = redis.Redis(host='127.0.0.1', port=6379)
    _redis_connection.set('qtt_abort_running_measurement', 0)
except BaseException:
    _redis_connection = None


def _abort_measurement(value=None):
    """ Return True if the currently running measurement should be aborted """
    if _redis_connection is None:
        return 0
    if value is not None:
        _redis_connection.set('qtt_abort_running_measurement', value)

    v = _redis_connection.get('qtt_abort_running_measurement')
    if v is None:
        v = 0
    return int(v)


def reset_abort(value=0):
    """ reset qtt_abort_running_measurement """
    _redis_connection.set('qtt_abort_running_measurement', value)


def _redisStrValue(var='qtt_live_value1'):
    """ Return live control value retrieved from redis server
        and convert to string """
    if _redis_connection is None:
        return 0
    v = _redis_connection.get(var)
    return v.decode('utf-8')


def _redisStrSet(value, var='qtt_live_value1'):
    """ Set live control value on redis server """
    _redis_connection.set(var, value)


liveValue = _redisStrValue
liveValueSet = _redisStrSet

abort_measurements = _abort_measurement  # type: ignore

# %% Override default location formatter

FormatLocation.default_fmt = '{date}/{time}_{name}_{label}'
qcodes_loop.data.data_set.DataSet.location_provider = FormatLocation(
    fmt='{date}/{time}_{name}_{label}', record={'name': 'qtt', 'label': 'generic'})


def set_location_name(name, verbose=1):
    if verbose:
        print('setting location name tag to %s' % name)
    qcodes_loop.data.data_set.DataSet.location_provider.base_record['name'] = name
# %%


def _copy_to_str(x, memo=None):
    return str(x)


def _setstate(self, d):
    self._short_name = d
    self._instrument = None

    def _get():
        print('instrument %s was serialized, no get available' % self.name)
        raise Exception('no get function defined')

    self.get = _get


# black magic to make qcodes objects work with deepcopy
for c in [Parameter, Instrument, ManualParameter, Station]:
    copy._deepcopy_dispatch[c] = _copy_to_str  # type: ignore

# make a qcodes instrument pickable
qcodes.Parameter.__getstate__ = _copy_to_str  # type: ignore
qcodes.Parameter.__setstate__ = _setstate  # type: ignore


# %% Enhance the qcodes functionality

try:
    from qcodes_loop.plots.pyqtgraph import QtPlot
    from qtpy import QtWidgets
    from qtpy.QtCore import Qt

    def _qtt_keyPressEvent(self, e):
        ''' Patch to add a callback to the QtPlot figure window '''
        if e.key() == Qt.Key_P:
            print('key P pressed: copy figure window to powerpoint')
            qtt.utilities.tools.addPPTslide(fig=self)
        super(QtPlot, self).keyPressEvent(e)

    # update the keypress callback function
    QtPlot.keyPressEvent = _qtt_keyPressEvent  # type: ignore
except BaseException:
    pass

# %% Enhance the qcodes functionality

try:
    import pyqtgraph as pg

    def _copyToClipboard(self):
        ''' Copy the current image to a the system clipboard '''
        app = pg.mkQApp()
        clipboard = app.clipboard()
        clipboard.setPixmap(self.win.grab())

    QtPlot.copyToClipboard = _copyToClipboard  # type: ignore
except BaseException:
    pass
