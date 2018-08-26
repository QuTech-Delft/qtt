""" QuTech Tuning

The QTT package contains functionality for the tuning and calibration of spin-qubits. The package is
divided into subpacakges:

    - Measurements: functionality to perform measurements on devices
    - Algorithms: functionality to analyse measurements 
    - Simulation: contains simulations of quantom dot systems
    - Tools: misc tools
    - Gui: Several gui element for visualization of data
    - Instrument drivers: contains QCoDeS drivers for various instruments

For more information see https://github.com/VandersypenQutech/qtt

"""
# flake8: noqa (we don't need the "<...> imported but unused" error)

import copy
import warnings
import importlib
import distutils
import distutils.version

import qcodes
import qtt.utilities.tools
import qtt.data
import qtt.algorithms
import qtt.measurements
import qtt.exceptions

from qtt.version import __version__
from qtt.measurements.storage import save_state, load_state

try:
    import pyqtgraph
    import qtt.gui.live_plotting
    import qtt.gui.parameterviewer
    from qtt.gui.parameterviewer import createParameterWidget
    
    from qtt.gui.dataviewer import DataViewer
except ImportError:
    # no gui available
    warnings.warn('pyqtgraph could not be imported, gui elements not available')
    

#%% Check packages


def check_version(version, module=qcodes, optional = False):
    """ Check whether a module has the corret version """
    if isinstance(module, str):
        try:
            m = importlib.import_module(module)
            module = m
        except ModuleNotFoundError:
            if optional:
                warnings.warn('optional package %s is not available' % module, qtt.exceptions.MissingOptionalPackageWarning)
                return
            else:
                raise Exception('could not load module %s' % module)
            
    mversion = getattr(module, '__version__', None)
    if mversion is None:
        raise Exception(' module %s has no __version__ attribute' % (module,))

            
    if distutils.version.StrictVersion(mversion) < distutils.version.StrictVersion(version):
        if optional:
            warnings.warn('package %s has incorrect version' % module, qtt.exceptions.PackageVersionWarning)
        else:
            raise Exception(' from %s need version %s (version is %s)' % (module, version, mversion))

# we make an explicit check on versions, since people often upgrade their installation without upgrading the required packages
check_version('1.0', 'qtpy')
check_version('0.18', 'scipy')
check_version('0.1', 'colorama')
check_version('0.1', 'redis', optional=True)
check_version('0.1.7', qcodes) # version of qcodes required


#%% Load often used constructions

from qtt.live_plotting import start_measurement_control


@qtt.utilities.tools.rdeprecated(expire='Aug 1 2018')
def start_dataviewer():
    from qtt.gui.dataviewer import DataViewer
    dv = DataViewer()
    dv.show()
    return dv


#%% Add hook to abort measurement

# connect to redis server
_redis_connection = None
try:
    import redis
    _redis_connection = redis.Redis(host='127.0.0.1', port=6379)
    _redis_connection.set('qtt_abort_running_measurement', 0)
except:
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


def reset_abort(value = 0):
    """ reset qtt_abort_running_measurement """
    _redis_connection.set('qtt_abort_running_measurement', value)

def _redisStrValue(var = 'qtt_live_value1'):
    """ Return live control value retrieved from redis server 
        and convert to string """
    if _redis_connection is None:
        return 0
    v = _redis_connection.get(var)
    return v.decode('utf-8')

def _redisStrSet(value, var = 'qtt_live_value1'):
    """ Set live control value on redis server """
    _redis_connection.set(var, value)

liveValue = _redisStrValue
liveValueSet = _redisStrSet

abort_measurements = _abort_measurement

# patch the qcodes abort function
qcodes.loops.abort_measurements = _abort_measurement

qtt._dummy_mc = []

#%% Override default location formatter

from qcodes.data.location import FormatLocation
FormatLocation.default_fmt = '{date}/{time}_{name}_{label}'
qcodes.DataSet.location_provider = FormatLocation(
    fmt='{date}/{time}_{name}_{label}', record={'name': 'qtt', 'label': 'generic'})


def set_location_name(name, verbose=1):
    if verbose:
        print('setting location name tag to %s' % name)
    qcodes.DataSet.location_provider.base_record['name'] = name
#%%

def _copy_to_str(x, memo):
    return str(x)


# black magic to make qcodes objects work with deepcopy
from qcodes import Parameter, Instrument, StandardParameter, ManualParameter, Station
for c in [Parameter, Instrument, StandardParameter, ManualParameter, Station]:
    copy._deepcopy_dispatch[c] = _copy_to_str

#qcodes.DataArray.omit_delegate_attrs += ['__deepcopy__']

# make a qcodes instrument pickable
qcodes.Instrument.__getstate__ = lambda self: str(self)
qcodes.Parameter.__getstate__ = lambda self: str(self)


def _setstate(self, d):
    self.name = d
    self._instrument = None
    
    def _get():
        print('instrument %s was serialized, no get available' % self.name)
        raise Exception('no get function defined')
    self.get = _get
    

qcodes.Instrument.__setstate__ = _setstate
qcodes.Parameter.__setstate__ = _setstate

#%% Enhance the qcodes functionality

try:
    from qtpy.QtCore import Qt
    from qtpy import QtWidgets
    from qcodes.plots.pyqtgraph import QtPlot

    def _qtt_keyPressEvent(self, e):
        ''' Patch to add a callback to the QtPlot figure window '''
        if e.key() == Qt.Key_P:
            print('key P pressed: copy figure window to powerpoint')
            qtt.utilities.tools.addPPTslide(fig=self)
        super(QtPlot, self).keyPressEvent(e)

    # update the keypress callback function
    QtPlot.keyPressEvent = _qtt_keyPressEvent
except:
    pass

#%% Enhance the qcodes functionality

try:
    import pyqtgraph as pg
    
    
    def _copyToClipboard(self):
        ''' Copy the current image to a the system clipboard '''
        app = pg.mkQApp()
        clipboard = app.clipboard()
        clipboard.setPixmap(self.win.grab())
    
    
    QtPlot.copyToClipboard = _copyToClipboard
except:
    pass
