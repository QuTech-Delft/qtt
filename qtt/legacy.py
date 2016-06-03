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

import pickle

def loadpickle(pkl_file):
    """ Load objects from file """
    try:    
        output = open(pkl_file, 'rb')
        data2 = pickle.load(output)
        output.close()
    except:
        if sys.version_info.major>=3:
            # if pickle file was saved in python2 we might fix issues with a different encoding
            output = open(pkl_file, 'rb')
            data2 = pickle.load(output, encoding='latin')
            #pickle.load(pkl_file, fix_imports=True, encoding="ASCII", errors="strict")
            output.close()
        else:
            data2=None
    return data2
    
def load_qt(fname):
    """ Load qtlab style file """
    alldata = loadpickle(fname)
    if isinstance(alldata, tuple):
        alldata = alldata[0]
    return alldata
