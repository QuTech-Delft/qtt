# -*- coding: utf-8 -*-
""" Example script to show QTT capabilities

@author: eendebakpt
"""

#%% Load packages
from imp import reload
import sys
import os
import numpy as np
import matplotlib
import time;
import pyqtgraph as pg
import tempfile

import qcodes
from qcodes.plots.pyqtgraph import QtPlot
from qcodes.plots.qcmatplotlib import MatPlot

import qtt
from qtt.parameterviewer import createParameterWidget
from qtt.algorithms.gatesweep import analyseGateSweep

if __name__ == '__main__':
    datadir = os.path.join(tempfile.tempdir, 'qdata')
    qcodes.DataSet.default_io = qcodes.DiskIO(datadir)
    

#%% Create a virtual model for testing
#
# The model resembles the spin-qubit dot setup. The hardware consists of a virtual
# keithley, IVVI racks and a virtual gates object

import virtualDot

if __name__ == '__main__':
    nr_dots = 2
    station = virtualDot.initialize(server_name=None, nr_dots=nr_dots)

    keithley1 = station.keithley1
    keithley3 = station.keithley3

    # virtual gates for the model
    gates = station.gates
    model = station.model

#%% Setup measurement windows


if __name__ == '__main__':
    mwindows = qtt.tools.setupMeasurementWindows(station, create_parameter_widget=False)
    pv = createParameterWidget([gates, ])
    plotQ = mwindows['plotwindow']
    
    logviewer = qtt.gui.dataviewer.DataViewer()
    logviewer.show()

#%% Read out instruments
if __name__ == '__main__':
    print('value: %f' % keithley3.readnext())
    snapshotdata = station.snapshot()


#%% Simple 1D scan loop

if __name__ == '__main__':
    scanjob = dict({'sweepdata': dict({'param': 'R', 'start': -500, 'end': 1, 'step': .8, 'wait_time': 5e-3}), 'minstrument': [keithley3.amplitude]})
    data1d = qtt.scans.scan1D(station, scanjob, location=None, verbose=1)


#%% Print the scanned data

if __name__ == '__main__':
    print( data1d.default_parameter_name() )


#%% Make a 2D scan
if __name__ == '__main__':

    reload(qtt.scans)
    start=-500
    scanjob = dict({'sweepdata': dict({'param': 'R', 'start': start, 'end': start+400, 'step': 4.}), 'minstrument': ['keithley1'], 'wait_time': 0.})
    scanjob['stepdata'] = dict({'param': 'L', 'start': start, 'end': start+400, 'step': 5.})
    data = qtt.scans.scan2D(station, scanjob, liveplotwindow=plotQ)

    #plotQ.clear(); plotQ.add(qtt.scans.getDefaultParameter(data))

#%% Fit 1D pinch-off scan:

if __name__ == '__main__':
    adata = analyseGateSweep(data1d, fig=100)

#%% Fit 2D cross
if __name__ == '__main__':
    from qtt.legacy import analyse2dot
    qtt.scans.plotData(data, fig=30)

    pt, resultsfine = analyse2dot(data, fig=300, efig=400, istep=1, verbose=2)

    
#%% Send data to powerpoint
if __name__ == '__main__':
    print('add copy data to Powerpoint use the following:')
    print('   qtt.tools.addPPT_dataset(data);')
    if 0:
        qtt.tools.addPPT_dataset(data)



