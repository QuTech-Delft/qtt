# set up the qtt namespace
# flake8: noqa (we don't need the "<...> imported but unused" error)

import matplotlib
if 0:
    try:
        if qtpy.API_NAME == 'PyQt4 (API v2)':
            matplotlib.use('Qt4Agg')
    except:
        pass
import qtt.live
import qtt.tools

import qtt.data
import qtt.algorithms

# todo: remove import *
from qtt.tools import cfigure, plot2Dline
from qtt.data import *
from qtt.algorithms import *
from qtt.algorithms.functions import logistic


#%% Enhance the qcodes functionality

try:
    import qtpy
except:
    pass
try:
    from qtpy.QtCore import Qt
    from qcodes.plots.pyqtgraph import QtPlot

    def keyPressEvent(self, e):
        ''' Patch to add a callback to the QtPlot figure window '''
        if e.key() == Qt.Key_P:
            print('key P pressed: copy figure window to powerpoint')
            qtt.tools.addPPTslide(txt='', fig=self)
        super(QtPlot, self).keyPressEvent(e)

    QtPlot.keyPressEvent = keyPressEvent  # update the keypress callback function
except:
    pass
