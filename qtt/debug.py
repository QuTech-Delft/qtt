#%% Tools for debugging code

import numpy as np
import scipy
import matplotlib
import sys
import os
import logging
import qcodes
import pickle

# explicit import
from qcodes.plots.pyqtgraph import QtPlot
from qcodes.plots.qcmatplotlib import MatPlot

from qtt import pmatlab
import tempfile

#%% Debugging


def dumpstring(txt, tag='dump'):
    with open(os.path.join(tempfile.tempdir, 'qtt-%s.txt' % tag), 'a+t') as fid:
        fid.write(txt + '\n')

#%% Decorator for generic function


import inspect
import functools
import tempfile
import time
import datetime


def functioncalldecorator(f, name=None):
    """ Decorate a function to log input and output arguments """
    if name is None:
        try:
            # try to get name from parent class
            c = f.__self__
            name = c.name
        except:
            name = 'none'

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        dstr = str(datetime.datetime.now())
        clock = time.perf_counter()
        r = f(*args, **kwargs)
        ss = 'function %s: %.6f, %s\n' % (name, clock, dstr) + 'function %s: arguments %s, %s' % (name, args, kwargs)
        dumpstring(ss + '\n' + 'function %s: output %s\n' % (name, r, ), tag='functionlog')
        #print ('method %s: output %s' % (f, r) )
        return r
    return wrapped

#%%


def logInstrument(instrument):
    """ Decorate all parameters of an instrument with logging methods """
    for k in instrument.parameters:
        p = instrument.parameters[k]
        if p.has_get:
            print('decorate %s' % p)
            p.get = functioncalldecorator(p.get, '%s.get' % p.name)
        if p.has_get and p.has_set:
            p.set = functioncalldecorator(p.set, '%s.set' % p.name)

#%%
import time
if __name__ == '__main__':
    for ii in range(10000):
        # print(datetime.dtetime.now())
        #print( time.time())
        # print(time.process_time())
        # print(time.clock())
        print(time.perf_counter())
