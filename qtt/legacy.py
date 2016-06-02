import qtpy
#print(qtpy.API_NAME)

import numpy as np
import scipy
import matplotlib
import sys, os
import logging
import qcodes

# explicit import
from qcodes.plots.pyqtgraph import QtPlot
from qcodes.plots.qcmatplotlib import MatPlot


# should be removed later
#from pmatlab import tilefigs


#%%

def getODbalancepoint(od):
    bp = od['balancepoint']
    if 'balancepointfine' in od:
        bp = od['balancepointfine']
    return bp


