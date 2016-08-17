# set up the qtt namespace
# flake8: noqa (we don't need the "<...> imported but unused" error)

import qtpy
import matplotlib
try:
    print(qtpy.API_NAME)
    if qtpy.API_NAME=='PyQt4 (API v2)':
        matplotlib.use('Qt4Agg')
except:
    pass
import qtt.live
import qtt.tools

# todo: remove import *
from qtt.tools import *
from qtt.data import *
from qtt.algorithms import *

from qcodes.plots.pyqtgraph import QtPlot

#%% Enhance the qcodes functionality

from qtpy.QtCore import Qt

def keyPressEvent(self, e):
    ''' Patch to add a callback to the QtPlot figure window '''
    if e.key() == Qt.Key_P:
        print('key P pressed: copy figure window to powerpoint')
        qtt.tools.addPPTslide(txt='', fig=self)
    super(QtPlot, self).keyPressEvent(e)
        
QtPlot.keyPressEvent=keyPressEvent # update the keypress callback function
