# set up the qtt namespace
# flake8: noqa (we don't need the "<...> imported but unused" error)

import copy
import matplotlib
import qcodes

import qtt.live
import qtt.tools

import qtt.data
import qtt.algorithms

import distutils.version

# todo: remove import *
from qtt.tools import cfigure, plot2Dline
from qtt.data import *
from qtt.algorithms import *
from qtt.algorithms.functions import logistic

#%%

def start_dataviewer():
    from qtt.gui.dataviewer import DataViewer
    dv = DataViewer()
    dv.show()
    return dv
from qtt.loggingGUI import installZMQlogger

_qversion = '0.1.2' # version of qcodes required
if distutils.version.StrictVersion(qcodes.__version__) < distutils.version.StrictVersion(_qversion):
    raise Exception('qtt needs qcodes version%s' % _qversion)

#%% Override default location formatter

from qcodes.data.location import FormatLocation
qcodes.DataSet.location_provider = FormatLocation(fmt='{date}/{time}_{name}_{label}', record={'name':'qtt', 'label': 'generic'})

def set_location_name(name, verbose=1):
    if verbose:
        print('setting location name tag to %s'% name)
    qcodes.DataSet.location_provider.base_record['name']=name
#%%

def _copy_to_str(x, memo):
    return str( x )

# black magic to make qcodes objects work with deepcopy
from qcodes import Parameter, Instrument, StandardParameter, ManualParameter
for c in [ Parameter, Instrument, StandardParameter, ManualParameter]:
    copy._deepcopy_dispatch[c] = _copy_to_str

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
