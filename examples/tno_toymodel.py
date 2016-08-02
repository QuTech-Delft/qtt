# -*- coding: utf-8 -*-
""" Toy model to test TNO algorithms with Qcodes

@author: eendebakpt
"""

#%% Load packages
from imp import reload
import math
import sys,os
import numpy as np
import time
import pdb
import multiprocessing as mp
if __name__=='__main__':
    try:
        mp.set_start_method('spawn')
    except:
        pass
    

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s: %(message)s  (%(filename)s:%(lineno)d)', )

import qcodes
import qcodes as qc
from qcodes import Instrument, MockInstrument, Parameter, Loop, DataArray
from qcodes.utils.validators import Numbers
import pickle
#import matplotlib.pyplot

if __name__=='__main__':
    l = logging.getLogger()
    l.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s (%(filename)s:%(lineno)d)')
    l.handlers[0].setFormatter(formatter)

import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot
matplotlib.pyplot.ion()

import pyqtgraph
import qtt

if __name__=='__main__':
    [ x.terminate() for x in qc.active_children() if x.name in ['dummymodel', 'ivvi1', 'ivvi2', 'AMockInsts'] ]

import virtualV2; # reload(virtualV2)
import qtt.qtt_toymodel; reload(qtt.qtt_toymodel)
from qtt.qtt_toymodel import FourdotModel, VirtualIVVI, VirtualMeter, logTest

#%% Create a virtual model for testing
#
# The model resembles the 4-dot setup. The hardware consists of a virtual
# keithley, 2 virtual IVVI racks

if __name__=='__main__':


    server_name='testv%d' % np.random.randint(1000) # needs to be set for background loops to work
    server_name=None
    virtualV2.initialize(server_name=server_name)
    #virtualV2.initialize(server_name='virtualV2'+time.strftime("%H.%M.%S"))
    
    
    keithley1 = virtualV2.keithley1
    keithley2 = virtualV2.keithley2
    keithley3 = virtualV2.keithley3
    ivvi1 = virtualV2.ivvi1
    
    # virtual gates for the model
    gates=virtualV2.gates
    
    model=virtualV2.model
    
    #%%
    logging.warning('test IVVI...')
    virtualV2.ivvi1.dac1.set(300)
    print('get P1: %f'  % (virtualV2.ivvi1.dac1.get(), ) )
    
#%%
    try:
        dot = gates.visualize()    
        #dot.view()
        qtt.showDotGraph(dot, fig=12)
        qtt.tilefigs(12, [1,2])
    except:
        pass
    
    #%%
    
    gate_boundaries=virtualV2.V2boundaries()
    

#%%
#import qcodes.instrument_drivers.QuTech.TimeStamp
from qtt.instrument_drivers.TimeStamp import TimeStampInstrument
if __name__=='__main__':

    ts = TimeStampInstrument(name='TimeStamp')
    
    
    station = virtualV2.getStation()
    station.set_measurement(keithley3.amplitude, ts.timestamp)



#%%
if __name__=='__main__':

    dd = station.snapshot()
    print(dd)

#%%

import qtt.scans
import platform

if __name__=='__main__':
    import pyqtgraph as pg
    qtapp=pg.mkQApp()
    
    qtt.pythonVersion()
    
    qdatadir = os.path.join(os.path.expanduser('~'), 'tmp', 'qdata')
    qcodes.DataSet.default_io = qcodes.DiskIO(qdatadir)
    #qcodes.DataSet.default_io = qcodes.DiskIO('/home/eendebakpt/tmp/qdata')
    mwindows=qtt.setupMeasurementWindows(station)
    mwindows['parameterviewer'].callbacklist.append( mwindows['plotwindow'].update )
    plotQ=mwindows['plotwindow']
    
    qtt.live.mwindows=mwindows

#%%
if __name__=='__main__':
    print('value: %f'  % keithley3.readnext() )
    
    #%%
    snapshotdata = station.snapshot()


#%%

import inspect

def showCaller(offset=1):
    st=inspect.stack()
    print('function %s: caller: %s:%s name: %s' % (st[offset][3], st[offset+1][1], st[offset+1][2], st[offset+1][3] ) )



#%% Simple 1D scan loop


from qtt.scans import scan1D



        
#%%
if 0:
    import h5py
    
    qcodes.DataSet.default_io
    
    import qcodes.tests
    import qcodes.tests.data_mocks
    data=qcodes.tests.data_mocks.DataSet1D()
    
    data.formatter
    
    from qcodes.data.hdf5_format import HDF5Format
    from qcodes.data.gnuplot_format import GNUPlotFormat
    
    formatter = HDF5Format()
    formatter = GNUPlotFormat()
    
    formatter.write(data)

        

#%%

#FIXME: set everything under __name__

if __name__=='__main__':
    scanjob = dict( {'sweepdata': dict({'gate': 'R', 'start': -500, 'end': 1, 'step': .5}), 'instrument': [keithley3.amplitude], 'delay': .000})
    data = qtt.scans.scan1D(scanjob, station, location=None, background=False)


    data.sync(); # data.arrays


#data = scan1D(scanjob, station, location='testsweep3', background=True)

#%

#reload(qcodes); reload(qc); plotQ=None

#%%

if __name__=='__main__':
    p=qtt.scans.getDefaultParameter(data)
    print(p)

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
from imp import reload
if __name__=='__main__':
    reload(qcodes.plots)
    reload(qtt.scans)
    
    scanjob = dict( {'sweepdata': dict({'gate': 'P1', 'start': -230, 'end': 160, 'step': 3.}), 'instrument': [keithley1.amplitude, keithley3.amplitude], 'delay': 0.})
    scanjob = dict( {'sweepdata': dict({'gate': 'P1', 'start': -230, 'end': 160, 'step': 2.}), 'instrument': [keithley1.amplitude], 'delay': 0.})
    #scanjob = dict( {'sweepdata': dict({'gate': 'P1', 'start': -230, 'end': 160, 'step': 6.}), 'instrument': [gates.L], 'delay': 0.})
    scanjob['stepdata']=dict({'gate': 'P3', 'start': -190, 'end': 120, 'step': 2.})
    data = qtt.scans.scan2D(station, scanjob, background=False)

    plotQ.clear(); plotQ.add(qtt.scans.getDefaultParameter(data))

#%%

if __name__=='__main__':
    
    STOP


#%% Check live plotting
if __name__=='__main__':

    data = scan1D(scanjob, station, location=None, background=False)

#%% Extend model (testing area)

if __name__=='__main__' and 0:

    model=virtualV2.model
    
    import qtt.simulation.dotsystem
    reload(qtt.simulation.dotsystem)
    from qtt.simulation.dotsystem import DotSystem, FourDot, GateTransform
    
    ds=FourDot()
    
    
    for ii in range(ds.ndots):
        setattr(ds, 'osC%d' % ( ii+1), 35)
    for ii in range(ds.ndots-1):
        setattr(ds, 'isC%d' % (ii+1), 3)
    
    targetnames=['det%d' % (i+1) for i in range(4)]
    sourcenames=['P%d' % (i+1) for i in range(4)]
    
    Vmatrix = qtt.simulation.dotsystem.defaultVmatrix(n=4)
    
    
    gate_transform = GateTransform(Vmatrix, sourcenames, targetnames)
    
    # fixme: does NOT work, we need to delegate the function to the network...
    
    model.fourdot = ds
    model.gate_transform=gate_transform
    
    self=model
    def computeSD(self, usediag=True, verbose=0):
        gv=[gates.get(g) for g in sourcenames ] 
        tv=gate_transform.transformGateScan(gv)
        for k, val in tv.items():
            if verbose:
                print('compudateSD: %d, %f'  % (k,val) )
            setattr(ds, k, val)
        ds.makeH()
        ds.solveH(usediag=usediag)
        ret = ds.OCC
    
        return ret
    
    
    tmp=computeSD(self)
    print(tmp)

    #model._data=model.get_attribute('_data')
    
#%%  

    
#%%

    
#%%
if __name__=='__main__':

    stepvalues=gates.R[0:100:1]
    data = qc.Loop(stepvalues, delay=.01, progress_interval=1).run(background=False)




if __name__=='__main__':

    plotQ=qc.QtPlot(data.amplitude_0)
    plotQ.win.setGeometry(1920, 10, 800,600)
    plotQ=qc.QtPlot(data.amplitude_1)
    plotQ.win.setGeometry(1920+800, 10, 800,600)


#%% Go!
if __name__=='__main__':

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


if __name__=='__main__':

    scanjob = dict({'sweepdata': dict({'gate': 'R', 'start': -220, 'end': 220, 'step': 2.5}), 'delay': .01})
    #scanjob = dict({'sweepdata': dict({'gate': 'R', 'start': 220, 'end': -220, 'step': -2.5}), 'delay': .01})

#%% Log file viewer
if __name__=='__main__':

    dd=os.listdir(qcodes.DataSet.default_io.base_location)

    
    


#%%

if __name__=='__main__':
    STOP

#%%
if __name__=='__main__':

    scanjob = dict({'sweepdata': dict({'gate': 'R', 'start': 220, 'end': -220, 'step': 3.5}), 'delay': .01})
    data = scan1D(scanjob, station, location=None, qcodesplot=plotQ)
    print(data)



if __name__=='__main__':

    dd=data
    adata=qtt.analyseGateSweep(dd, fig=10, verbose=2)
    qtt.tilefigs(10, [2,2])


#%% ########################################################################

#%% Load and analyse data

if 0:
    def load_data(location=None, **kwargs):
           if isinstance(location, int):
               dd=os.listdir(qcodes.DataSet.default_io.base_location)
               lastdate=sorted(dd)[-1]
               dd=sorted(os.listdir(os.path.join(qcodes.DataSet.default_io.base_location, lastdate) ))[::-1]
               location=os.path.join(lastdate, dd[location])
               #location=location.replace('.dat', '')
               logging.info('location: %s' % location)
           return qc.load_data(location, **kwargs)
           
           
    data=load_data(location=0)
    
    #qc.MatPlot(data.amplitude, fig=10)
    
    import pmatlab
    
    qc.MatPlot(data.amplitude, subplots=dict({'num':10}) )
    pmatlab.tilefigs(10,[2,2])


