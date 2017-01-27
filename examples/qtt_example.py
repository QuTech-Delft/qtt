# -*- coding: utf-8 -*-
""" Example script to show QTT capabilities

@author: eendebakpt
"""

#%% Load packages
from imp import reload
import sys
import os
import numpy as np
import time; time.sleep(1e-3)
import pyqtgraph as pg
import tempfile

import multiprocessing as mp
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except:
        pass

import qcodes
import qcodes as qc
import matplotlib
from qcodes.plots.pyqtgraph import QtPlot
from qcodes.plots.qcmatplotlib import MatPlot


import qtt
import qtt.scans
import qtt.qtt_toymodel
import qtt.live
from qtt.scans import scan1D
import qtt.gui.dataviewer

# taskkill /F /IM python.exe

if __name__ == '__main__':
    datadir = os.path.join(tempfile.tempdir, 'qdata')
    qcodes.DataSet.default_io = qcodes.DiskIO(datadir)
    
    
    try:
        from qcodes.data.hdf5_format import HDF5Format
        qcodes.DataSet.default_formatter = HDF5Format()
    except:
        pass

#%% Create a virtual model for testing
#
# The model resembles the spin-qubit dot setup. The hardware consists of a virtual
# keithley, IVVI racks and a virtual gates object

import virtualDot

if __name__ == '__main__':
    nr_dots = 2

    #virtualDot.close()
    server_name = 'testv%d' % np.random.randint(1000)  # needs to be set for background loops to work
    server_name = None
    station = virtualDot.initialize(server_name=server_name, nr_dots=nr_dots)

    keithley1 = station.keithley1
    keithley3 = station.keithley3
    ivvi1 = station.ivvi1

    # virtual gates for the model
    gates = station.gates
    model = station.model

#%% Setup measurement windows


if __name__ == '__main__':
    qtapp = pg.mkQApp()

    qdatadir = os.path.join(os.path.expanduser('~'), 'tmp', 'qdata')
    qcodes.DataSet.default_io = qcodes.DiskIO(qdatadir)
    mwindows = qtt.tools.setupMeasurementWindows(station, create_parameter_widget=False)
    if mwindows['parameterview'] is not None:
        mwindows['parameterviewer'].callbacklist.append(mwindows['plotwindow'].update)
    from qtt.parameterviewer import createParameterWidgetRemote, createParameterWidget
    if server_name is None:
        createParameterWidget([gates, ])
    else:
        createParameterWidgetRemote([gates, ])
    plotQ = mwindows['plotwindow']
    qtt.live.liveplotwindow = plotQ
    qtt.live.mwindows = mwindows

    logviewer = qtt.gui.dataviewer.DataViewer()
    logviewer.show()


#%% Read out instruments
if __name__ == '__main__':
    print('value: %f' % keithley3.readnext())
    snapshotdata = station.snapshot()


#%% Simple 1D scan loop


if __name__ == '__main__':
    reload(qtt.scans)
    scanjob = dict({'sweepdata': dict({'gate': 'R', 'start': -500, 'end': 1, 'step': 1.}), 'instrument': [keithley3.amplitude], 'wait_time': .000})
    data1d = qtt.scans.scan1D(scanjob, station, location=None, background=False, verbose=2)

    data1d.sync()  # data.arrays


#%% Print the scanned data

if __name__ == '__main__':
    p = qtt.scans.getDefaultParameter(data1d)
    print(p)


#%% Make a 2D scan
if __name__ == '__main__':

    reload(qtt.scans)
    start=-500
    scanjob = dict({'sweepdata': dict({'gate': 'R', 'start': start, 'end': start+500, 'step': 4.}), 'instrument': [keithley1.amplitude], 'wait_time': 0.})
    scanjob['stepdata'] = dict({'gate': 'L', 'start': start, 'end': start+500, 'step': 5.})
    data = qtt.scans.scan2D(station, scanjob, background=None, liveplotwindow=plotQ)

    #plotQ.clear(); plotQ.add(qtt.scans.getDefaultParameter(data))

#%% Fit 1D pinch-off scan:

from qtt.scans import analyseGateSweep

if __name__ == '__main__':
    adata = analyseGateSweep(data1d, fig=100)

#%% Fit 2D cross
if __name__ == '__main__':
    from qtt.legacy import *
    from qtt.legacy import analyse2dot
    qtt.scans.plotData(data, fig=30)

    pt, resultsfine = analyse2dot(data, fig=300, efig=400, istep=1, verbose=2)

    # TODO:
    #def onedotGetBalance(od, dd, verbose=1, fig=None, drawpoly=False, polylinewidth=2, linecolor='c'):

    
#%% Send data to powerpoint
if __name__ == '__main__':
    print('add copy data to Powerpoint use the following:')
    print('   qtt.tools.addPPT_dataset(data);')
    if 0:
        qtt.tools.addPPT_dataset(data)



