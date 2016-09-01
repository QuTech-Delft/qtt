# -*- coding: utf-8 -*-
""" Example script to show QTT capabilities

@author: eendebakpt
"""

#%% Load packages
from imp import reload
import sys,os
import numpy as np
import time
import pyqtgraph as pg
import tempfile

import multiprocessing as mp
if __name__=='__main__':
    try:
        mp.set_start_method('spawn')
    except:
        pass
    
import qcodes
import qcodes as qc
import matplotlib
matplotlib.use('Qt4Agg')

import qtt
import qtt.scans
import qtt.qtt_toymodel;
import qtt.live
from qtt.scans import scan1D


datadir = os.path.join(tempfile.tempdir, 'qdata')
qcodes.DataSet.default_io = qcodes.DiskIO(datadir)


try:
    from qcodes.data.hdf5_format import HDF5Format
    #qcodes.DataSet.default_formatter=HDF5Format()
except:
    pass

#%% Create a virtual model for testing
#
# The model resembles the spin-qubit dot setup. The hardware consists of a virtual
# keithley, IVVI racks and a virtual gates object

import virtualDot; 
#model = virtualDot.DotModel(name='dummymodel', server_name=None)

if __name__=='__main__':

    server_name='testv%d' % np.random.randint(1000) # needs to be set for background loops to work
    server_name=None
    station = virtualDot.initialize(server_name=server_name)    
    
    keithley1 = station.keithley1
    keithley3 = station.keithley3
    ivvi1 = station.ivvi1
    
    # virtual gates for the model
    gates=station.gates    
    model=station.model
    
    station.set_measurement(keithley3.amplitude)
        

#%% Setup measurement windows


if __name__=='__main__':
    qtapp=pg.mkQApp()
        
    qdatadir = os.path.join(os.path.expanduser('~'), 'tmp', 'qdata')
    qcodes.DataSet.default_io = qcodes.DiskIO(qdatadir)
    mwindows=qtt.setupMeasurementWindows(station)
    mwindows['parameterviewer'].callbacklist.append( mwindows['plotwindow'].update )
    plotQ=mwindows['plotwindow']
    qtt.live.liveplotwindow=plotQ
    qtt.live.mwindows=mwindows

    import qtt.gui
    import qtt.gui.dataviewer
    logviewer = qtt.gui.dataviewer.DataViewer()
    logviewer.show()


#%% Read out instruments 
if __name__=='__main__':
    print('value: %f'  % keithley3.readnext() )
    snapshotdata = station.snapshot()


#%% Simple 1D scan loop


if __name__=='__main__':
    scanjob = dict( {'sweepdata': dict({'gate': 'R', 'start': -500, 'end': 1, 'step': .2}), 'instrument': [keithley3.amplitude], 'delay': .000})
    data1d = qtt.scans.scan1D(scanjob, station, location=None, background=None)

    data1d.sync(); # data.arrays

    #data = scan1D(scanjob, station, location='testsweep3', background=False)


#%% Print the scanned data

if __name__=='__main__':
    p=qtt.scans.getDefaultParameter(data1d)
    print(p)


#%% Make a 2D scan
if __name__=='__main__':
    
    scanjob = dict( {'sweepdata': dict({'gate': 'R', 'start': -530, 'end': 160, 'step': 8.}), 'instrument': [keithley1.amplitude], 'delay': 0.})
    scanjob['stepdata']=dict({'gate': 'L', 'start': -340, 'end': 250, 'step': 10.})
    data = qtt.scans.scan2D(station, scanjob, background=None, liveplotwindow=plotQ)

    #plotQ.clear(); plotQ.add(qtt.scans.getDefaultParameter(data))

#%% Fit 1D pinch-off scan:

from qtt.scans import analyseGateSweep

if __name__=='__main__':
    adata = analyseGateSweep(data1d, fig=100)


#%% Send data to powerpoint


#%% DEBUGGING AND TESTING

#%%

if __name__=='__main__':
    
    gates.L.set(300); gates.R.set(300)
    print( model.keithley1_get('amplitude') )
    gates.L.set(-300.); gates.R.set(-300.)
    print( model.keithley1_get('amplitude') )
    print(model.my_get())
    
    STOP


#%% Check live plotting
if __name__=='__main__':

    data = scan1D(scanjob, station, location=None, background=False)


    
#%%
    
if __name__=='__main__':

    #plotQ = qc.MatPlot(data.amplitude)
    if plotQ is None:
        plotQ = qc.QtPlot(qtt.scans.getDefaultParameter(data), windowTitle='Live plot', remote=False)
        #plotQ.win.setGeometry(1920+360, 100, 800, 600)
        data.sync()    
        plotQ.update()
        mwindows['parameterviewer'].callbacklist.append( plotQ.update )
    else:
        data.sync()    
        plotQ.clear(); plotQ.add(qtt.scans.getDefaultParameter(data))
    
#%%
if __name__=='__main__':

    stepvalues=gates.R[0:100:1]
    data = qc.Loop(stepvalues, delay=.01, progress_interval=1).run(background=False)


#%% Go!
if __name__=='__main__':

    for ii in range(1):
        print('progress: fraction %.2f, %.1f seconds remaining' %
              qtt.timeProgress(data))
        plotQ.update()
        time.sleep(.1)





