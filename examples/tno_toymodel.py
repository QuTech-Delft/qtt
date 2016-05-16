# -*- coding: utf-8 -*-
""" Toy model to test TNO algorithms with Qcodes

@author: eendebakpt
"""

#%% Load packages
from imp import reload
import math
import sys,os
import numpy as np
import dill
import time
import pdb
import multiprocessing as mp

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s: %(message)s  (%(filename)s:%(lineno)d)', )

import qcodes
import qcodes as qc
from qcodes import Instrument, MockInstrument, Parameter, Loop, DataArray
from qcodes.utils.validators import Numbers

l = logging.getLogger()
l.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s (%(filename)s:%(lineno)d)')
l.handlers[0].setFormatter(formatter)

import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot

import pyqtgraph
import qtt

[ x.terminate() for x in qc.active_children() if x.name in ['dummymodel', 'ivvi1', 'ivvi2', 'AMockInsts'] ]


#%% Create a virtual model for testing
#
# The model resembles the 4-dot setup. The hardware consists of a virtual
# keithley, 2 virtual IVVI racks

import virtualV2; # reload(virtualV2)

virtualV2.initialize(server_name=None)
#virtualV2.initialize(server_name='virtualV2'+time.strftime("%H.%M.%S"))

import qtt.qtt_toymodel
from qtt.qtt_toymodel import DummyModel, VirtualIVVI, MockMeter, MockSource, logTest

keithley1 = virtualV2.keithley1
keithley2 = virtualV2.keithley2
keithley3 = virtualV2.keithley3

# virtual gates for the model
gates=virtualV2.gates

#%%
logging.warning('test IVVI...')
virtualV2.ivvi1.c1.set(300)
print('get P1: %f'  % (virtualV2.ivvi1.c1.get(), ) )


#%%

dot = gates.visualize()    
#dot.view()
qtt.showDotGraph(dot, fig=12)
qtt.tilefigs(12, [1,2])


#%%

gate_boundaries=virtualV2.V2boundaries()

#%%
#l.setLevel(logging.DEBUG)

for v in [-20, 0, 20, 40, 60]:
    gates.set_R(v)
    w = keithley3.readnext()
    print('v %f: w %f' % (v, w))


#%%
#import qcodes.instrument_drivers.QuTech.TimeStamp
from qtt.instrument_drivers.TimeStamp import TimeStampInstrument
ts = TimeStampInstrument(name='TimeStamp')


station = virtualV2.getStation()
station.set_measurement(keithley3.amplitude, ts.timestamp)



#%%

dd = station.snapshot()
print(dd)

#%%

import qtt.scans

qtt.pythonVersion()

qcodes.DataSet.default_io = qcodes.DiskIO('/home/eendebakpt/tmp/qdata')
mwindows=qtt.setupMeasurementWindows(station)
mwindows['parameterviewer'].callbacklist.append( mwindows['plotwindow'].update )
plotQ=mwindows['plotwindow']

qtt.scans.mwindows=mwindows

#%%


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


#%%
scanjob = dict( {'sweepdata': dict({'gate': 'R', 'start': -290, 'end': 160, 'step': 8.}), 'instrument': [keithley3.amplitude], 'delay': .01})
data = scan1D(scanjob, station, location='testsweep13', background=True)


data.sync(); data.arrays


#data = scan1D(scanjob, station, location='testsweep3', background=True)

#%

#reload(qcodes); reload(qc); plotQ=None

#plotQ = qc.MatPlot(data.amplitude)
if plotQ is None:
    plotQ = qc.QtPlot(data.amplitude, windowTitle='Live plot', remote=False)
    #plotQ.win.setGeometry(1920+360, 100, 800, 600)
    data.sync()    
    plotQ.update()
    mwindows['parameterviewer'].callbacklist.append( plotQ.update )
else:
    data.sync()    
    plotQ.clear(); plotQ.add(data.amplitude)
    

STOP

#%%
from imp import reload
reload(qcodes.plots)
reload(qtt.scans)

scanjob = dict( {'sweepdata': dict({'gate': 'R', 'start': -290, 'end': 160, 'step': 8.}), 'instrument': [keithley3.amplitude, keithley1.amplitude], 'delay': .01})
scanjob['stepdata']=dict({'gate': 'D1', 'start': -290, 'end': 160, 'step': 8.})
data = qtt.scans.scan2D(station, scanjob)


#qc.active_children()
#qc.halt_bg()
#plotQ.win.setGeometry(1920, 100, 800, 600)


#%%




#%% Go!

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



scanjob = dict({'sweepdata': dict({'gate': 'R', 'start': -220, 'end': 220, 'step': 2.5}), 'delay': .01})
#scanjob = dict({'sweepdata': dict({'gate': 'R', 'start': 220, 'end': -220, 'step': -2.5}), 'delay': .01})

#%% Log file viewer

dd=os.listdir(qcodes.DataSet.default_io.base_location)

    
    


#%%
STOP


#%%
scanjob = dict({'sweepdata': dict({'gate': 'R', 'start': 220, 'end': -220, 'step': 3.5}), 'delay': .01})
data = scan1D(scanjob, station, location=None, qcodesplot=plotQ)
print(data)


#%%
#
# TODO: code refactoring
# TODO: merge back into qutech/packages (for now)
# TODO: an check or scan direction
# TODO: clean up code

    

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


