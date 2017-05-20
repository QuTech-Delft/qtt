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
import time
import pyqtgraph as pg
import tempfile

import qcodes
from qcodes import QtPlot
from qcodes import MatPlot

import qtt
from qtt import createParameterWidget
from qtt.algorithms.gatesweep import analyseGateSweep
from qtt.measurements.scans import scanjob_t

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
    station = virtualDot.initialize(reinit=True, nr_dots=nr_dots)

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
    param_left=station.model.bottomgates[0]
    param_right=station.model.bottomgates[-1]
    scanjob = scanjob_t({'sweepdata': dict({'param': param_right, 'start': -500, 'end': 1, 'step': .8, 'wait_time': 5e-3}), 'minstrument': [keithley3.amplitude]})
    data1d = qtt.measurements.scans.scan1D(station, scanjob, location=None, verbose=1)


#%% Print the scanned data

if __name__ == '__main__':
    print(data1d.default_parameter_name())


#%% Make a 2D scan
if __name__ == '__main__':
    start = -500
    scanjob = scanjob_t({'sweepdata': dict({'param': param_right, 'start': start, 'end': start + 400, 'step': 4., 'wait_time': 0.}), 'minstrument': ['keithley1']})
    scanjob['stepdata'] = dict({'param': param_left, 'start': start, 'end': start + 400, 'step': 5.})
    data = qtt.measurements.scans.scan2D(station, scanjob, liveplotwindow=plotQ)

    gates.R.set(-300); gates.L.set(-300)
    gv=gates.allvalues()

    #plotQ.clear(); plotQ.add(qtt.scans.getDefaultParameter(data))

#%% Fit 1D pinch-off scan:

if __name__ == '__main__':
    adata = analyseGateSweep(data1d, fig=100)

#%% Fit 2D cross
if __name__ == '__main__':
    from qtt.legacy import analyse2dot
    qtt.measurements.scans.plotData(data, fig=30)

    pt, resultsfine = analyse2dot(data, fig=300, efig=400, istep=1, verbose=2)

    #gates.L.set(float(resultsfine['ptmv'][0]) )
    #gates.R.set(float( resultsfine['ptmv'][1]) )
    
 
    
#%% Make virtual gates

from collections import OrderedDict
from qtt.instrument_drivers.virtual_gates import virtual_gates

crosscap_map = OrderedDict((
('VP1', OrderedDict((('P1', 1), ('P2', 0.6), ('P3', 0)))),
('VP2', OrderedDict((('P1', 0.7), ('P2', 1), ('P3', 0.3)))),
('VP3', OrderedDict((('P1', 0), ('P2', 0), ('P3', 1))))
))
virts = virtual_gates(qtt.measurements.scans.instrumentName('vgates'), gates, crosscap_map)
gates.resetgates(gv, gv)

cc1= virts.VP1()
ccx= virts.VP2()
scanjob = scanjob_t({'sweepdata': dict({'param': virts.VP1, 'start': cc1-100, 'end': cc1 + 100, 'step': 4.}), 'minstrument': ['keithley1'], 'wait_time': 0.})
scanjob['stepdata'] = dict({'param': virts.VP2, 'start': ccx - 100, 'end': ccx +100, 'step': 2.})
data = qtt.measurements.scans.scan2D(station, scanjob, liveplotwindow=plotQ)
gates.resetgates(gv, gv)

#%% Send data to powerpoint
if __name__ == '__main__':
    print('add copy data to Powerpoint use the following:')
    print('   qtt.tools.addPPT_dataset(data);')
    if 0:
        qtt.tools.addPPT_dataset(data)

#%% Test objects

qtt.instrument_drivers.virtual_gates.test_virtual_gates()
qtt.measurements.scans.test_scan2D()