# -*- coding: utf-8 -*-
""" Toy model to test TNO algorithms with Qcodes

@author: eendebakpt
"""

#%% Load packages
from imp import reload
import math
import sys
import os
import numpy as np
import time
import pdb
import qtt.scans
import platform

import multiprocessing as mp
if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except:
        pass


import logging
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s: %(message)s  (%(filename)s:%(lineno)d)', )

import qcodes
import qcodes as qc
from qcodes import Instrument, Parameter, Loop, DataArray

if __name__ == '__main__':
    l = logging.getLogger()
    l.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s (%(filename)s:%(lineno)d)')
    l.handlers[0].setFormatter(formatter)

import matplotlib
import matplotlib.pyplot
if __name__ == '__main__':
    matplotlib.use('Qt4Agg')
    matplotlib.pyplot.ion()

import pyqtgraph
import qtt

if __name__ == '__main__':
    [x.terminate() for x in qc.active_children() if x.name in ['dummymodel', 'ivvi1', 'ivvi2']]

import virtualV2  # reload(virtualV2)
import qtt.qtt_toymodel  # reload(qtt.qtt_toymodel)
import qtt.live
from qtt.qtt_toymodel import FourdotModel
from qtt.scans import scan1D

import tempfile

import matplotlib.pyplot
if __name__ == '__main__':

    datadir = os.path.join(tempfile.tempdir, 'qdata')
    qcodes.DataSet.default_io = qcodes.DiskIO(datadir)


#%% Create a virtual model for testing
#
# The model resembles the 4-dot setup. The hardware consists of a virtual
# keithley, 2 virtual IVVI racks

if __name__ == '__main__':

    server_name = 'testv%d' % np.random.randint(1000)  # needs to be set for background loops to work
    # server_name=None
    virtualV2.initialize(server_name=server_name)

    keithley1 = virtualV2.keithley1
    keithley2 = virtualV2.keithley2
    keithley3 = virtualV2.keithley3
    ivvi1 = virtualV2.ivvi1

    # virtual gates for the model
    gates = virtualV2.gates
    model = virtualV2.model
    gate_boundaries = virtualV2.V2boundaries()

    station = virtualV2.getStation()
    station.set_measurement(keithley3.amplitude)

#%%
    try:
        dot = gates.visualize()
        # dot.view()
        qtt.showDotGraph(dot, fig=12)
        qtt.tilefigs(12, [1, 2])
    except:
        pass


#%% Setup measurement windows


if __name__ == '__main__':
    import pyqtgraph as pg
    qtapp = pg.mkQApp()

    qtt.pythonVersion()

    qdatadir = os.path.join(os.path.expanduser('~'), 'tmp', 'qdata')
    qcodes.DataSet.default_io = qcodes.DiskIO(qdatadir)
    mwindows = qtt.setupMeasurementWindows(station)
    mwindows['parameterviewer'].callbacklist.append(mwindows['plotwindow'].update)
    plotQ = mwindows['plotwindow']

    qtt.live.mwindows = mwindows

    import qcodes.tools.dataviewer
    logviewer = qcodes.tools.dataviewer.DataViewer()
    logviewer.show()


#%%
if __name__ == '__main__':
    print('value: %f' % keithley3.readnext())
    snapshotdata = station.snapshot()


#%% Simple 1D scan loop


if __name__ == '__main__':
    scanjob = dict({'sweepdata': dict({'gate': 'R', 'start': -500, 'end': 1, 'step': .5}), 'instrument': [keithley3.amplitude], 'delay': .000})
    data = qtt.scans.scan1D(scanjob, station, location=None, background=False)

    data.sync()  # data.arrays


#data = scan1D(scanjob, station, location='testsweep3', background=True)

#%%

if __name__ == '__main__':
    p = qtt.scans.getDefaultParameter(data)
    print(p)

#%%

if __name__ == '__main__':

    #plotQ = qc.MatPlot(data.amplitude)
    if plotQ is None:
        plotQ = qc.QtPlot(qtt.scans.getDefaultParameter(data), windowTitle='Live plot', remote=False)
        #plotQ.win.setGeometry(1920+360, 100, 800, 600)
        data.sync()
        plotQ.update()
        mwindows['parameterviewer'].callbacklist.append(plotQ.update)
    else:
        data.sync()
        plotQ.clear()
        plotQ.add(qtt.scans.getDefaultParameter(data))


#%%
from imp import reload
if __name__ == '__main__':
    reload(qcodes.plots)
    reload(qtt.scans)

    scanjob = dict({'sweepdata': dict({'gate': 'P1', 'start': -230, 'end': 160, 'step': 3.}), 'instrument': [keithley1.amplitude, keithley3.amplitude], 'delay': 0.})
    scanjob = dict({'sweepdata': dict({'gate': 'P1', 'start': -230, 'end': 160, 'step': 2.}), 'instrument': [keithley1.amplitude], 'delay': 0.})
    #scanjob = dict( {'sweepdata': dict({'gate': 'P1', 'start': -230, 'end': 160, 'step': 6.}), 'instrument': [gates.L], 'delay': 0.})
    scanjob['stepdata'] = dict({'gate': 'P3', 'start': -190, 'end': 120, 'step': 2.})
    data = qtt.scans.scan2D(station, scanjob, background=False)

    plotQ.clear()
    plotQ.add(qtt.scans.getDefaultParameter(data))

#%%

if __name__ == '__main__':

    STOP


#%% Check live plotting
if __name__ == '__main__':

    data = scan1D(scanjob, station, location=None, background=False)

#%% Extend model (testing area)

if __name__ == '__main__' and 0:

    model = virtualV2.model

    import qtt.simulation.dotsystem
    reload(qtt.simulation.dotsystem)
    from qtt.simulation.dotsystem import DotSystem, FourDot, GateTransform

    ds = FourDot()

    for ii in range(ds.ndots):
        setattr(ds, 'osC%d' % (ii + 1), 35)
    for ii in range(ds.ndots - 1):
        setattr(ds, 'isC%d' % (ii + 1), 3)

    targetnames = ['det%d' % (i + 1) for i in range(4)]
    sourcenames = ['P%d' % (i + 1) for i in range(4)]

    Vmatrix = qtt.simulation.dotsystem.defaultVmatrix(n=4)

    gate_transform = GateTransform(Vmatrix, sourcenames, targetnames)

    # fixme: does NOT work, we need to delegate the function to the network...

    model.fourdot = ds
    model.gate_transform = gate_transform

    self = model
    def computeSD(self, usediag=True, verbose=0):
        gv = [gates.get(g) for g in sourcenames]
        tv = gate_transform.transformGateScan(gv)
        for k, val in tv.items():
            if verbose:
                print('compudateSD: %d, %f' % (k, val))
            setattr(ds, k, val)
        ds.makeH()
        ds.solveH(usediag=usediag)
        ret = ds.OCC

        return ret

    tmp = computeSD(self)
    print(tmp)

    # model._data=model.get_attribute('_data')

#%%


#%%


#%%
if __name__ == '__main__':

    stepvalues = gates.R[0:100:1]
    data = qc.Loop(stepvalues, delay=.01, progress_interval=1).run(background=False)


#%% Go!
if __name__ == '__main__':

    for ii in range(1):
        print('progress: fraction %.2f, %.1f seconds remaining' %
              qtt.timeProgress(data))
        plotQ.update()
        time.sleep(.1)


#%%
if 0:
    scanjob = dict({'sweepdata': dict({'gate': 'R', 'start': -420, 'end': 220, 'step': 2.5}), 'delay': .01})
    data = scan1D(scanjob, station, location='testsweep4')
    plotQ.add(data.amplitude)

    #%%
    plotQ.add(np.array(data.amplitude) + .2)

    #%%

    datax = qc.DataSet('testsweep3', mode=qcodes.DataMode.LOCAL)

    fig = qc.MatPlot(datax.amplitude)

    import pmatlab
    pmatlab.tilefigs([fig.fig], [2, 2])


#%%
if __name__ == '__main__':

    scanjob = dict({'sweepdata': dict({'gate': 'R', 'start': 220, 'end': -220, 'step': 3.5}), 'delay': .01})
    data = scan1D(scanjob, station, location=None, qcodesplot=plotQ)
    print(data)


if __name__ == '__main__':

    dd = data
    adata = qtt.analyseGateSweep(dd, fig=10, verbose=2)
    qtt.tilefigs(10, [2, 2])


#%% ########################################################################
