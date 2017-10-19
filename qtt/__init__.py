# set up the qtt namespace
# flake8: noqa (we don't need the "<...> imported but unused" error)

import copy
import qcodes

import warnings
import qtt.live
import qtt.tools
import qtt.data
import qtt.algorithms
import qtt.measurements
import qtt.utilities.markup as markup

import distutils.version

from qtt.version import __version__

from qtt.tools import cfigure, plot2Dline
from qtt.data import *
from qtt.algorithms.functions import logistic
from qtt.measurements.storage import save_state, load_state 

import qtt.live_plotting
import qtt.gui.parameterviewer
from qtt.gui.parameterviewer import createParameterWidget

from qtt.gui.dataviewer import DataViewer

#%%

if qcodes.config['user'].get('deprecation_warnings', True):
    # enable deprecation warnings
    warnings.simplefilter("default", DeprecationWarning)

#%% Load often used constructions

from qtt.live_plotting import start_measurement_control

def start_dataviewer():
    from qtt.gui.dataviewer import DataViewer
    dv = DataViewer()
    dv.show()
    return dv
from qtt.loggingGUI import installZMQlogger

def check_version(version, module=qcodes):
    if distutils.version.StrictVersion(module.__version__) < distutils.version.StrictVersion(version):
        raise Exception(' from %s need version %s' % (module, version) )

_qversion = '0.1.3' # version of qcodes required
check_version(_qversion)

#%% Add hook to abort measurement

# connect to redis server
_redis_connection = None
try:
    import redis
    _redis_connection = redis.Redis(host='127.0.0.1', port=6379)
    _redis_connection.set('qtt_abort_running_measurement', 0)
except:
    _redis_connection = None
    
    pass

def _abort_measurement():
    """ Return True if the currently running measurement should be aborted """
    if _redis_connection is None:
        return 0
    v=_redis_connection.get('qtt_abort_running_measurement')
    if v is None:
        v=0
    return int(v)

abort_measurements = _abort_measurement
# patch the qcodes abort function
qcodes.loops.abort_measurements = _abort_measurement

qtt._dummy_mc = []

#%% Override default location formatter

from qcodes.data.location import FormatLocation
FormatLocation.default_fmt = '{date}/{time}_{name}_{label}'
qcodes.DataSet.location_provider = FormatLocation(fmt='{date}/{time}_{name}_{label}', record={'name':'qtt', 'label': 'generic'})

def set_location_name(name, verbose=1):
    if verbose:
        print('setting location name tag to %s'% name)
    qcodes.DataSet.location_provider.base_record['name']=name
#%%

def _copy_to_str(x, memo):
    return str( x )

# black magic to make qcodes objects work with deepcopy
from qcodes import Parameter, Instrument, StandardParameter, ManualParameter, Station
for c in [ Parameter, Instrument, StandardParameter, ManualParameter, Station]:
    copy._deepcopy_dispatch[c] = _copy_to_str


# make a qcodes instrument pickable 
qcodes.Instrument.__getstate__ = lambda self: str(self)
qcodes.Parameter.__getstate__ = lambda self: str(self)

def _setstate(self, d):
    self.name = d
    self._instrument=None

qcodes.Instrument.__setstate__ = _setstate
qcodes.Parameter.__setstate__ = _setstate
    
#%% Enhance the qcodes functionality

try:
    import qtpy
except:
    pass
try:
    from qtpy.QtCore import Qt
    from qtpy import QtWidgets
    from qcodes.plots.pyqtgraph import QtPlot

    def keyPressEvent(self, e):
        ''' Patch to add a callback to the QtPlot figure window '''
        if e.key() == Qt.Key_P:
            print('key P pressed: copy figure window to powerpoint')
            qtt.tools.addPPTslide(txt='', fig=self)
        super(QtPlot, self).keyPressEvent(e)

    # update the keypress callback function
    QtPlot.keyPressEvent = keyPressEvent
except:
    pass

#%%
import pyqtgraph as pg

def _copyToClipboard(self):
        ''' Copy the current image to a the system clipboard '''
        app = pg.mkQApp()
        clipboard = app.clipboard()
        clipboard.setPixmap(pg.QtGui.QPixmap.grabWidget(self))
        
QtPlot.copyToClipboard=_copyToClipboard        