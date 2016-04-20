# -*- coding: utf-8 -*-
""" Toy model to test TNO algorithms with Qcodes

@author: eendebakpt
"""

#%% Load packages
from imp import reload
import math
import sys
import numpy as np

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s: %(message)s (%(filename)s:%(lineno)d)', )


import qcodes
import qcodes as qc
from qcodes import Instrument, MockInstrument, Parameter, Loop, DataArray
from qcodes.utils.validators import Numbers


l = logging.getLogger()
# l.setLevel(logging.DEBUG)
l.setLevel(logging.INFO)

import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot

import pyqtgraph
import qtt
import qutechalgorithms

logging.warning('test')


[ x.terminate() for x in qc.active_children() if x.name in ['dummymodel', 'ivvi1', 'ivvi2'] ]

#%% Create a virtual model for testing
#
# The model resembles the 4-dot setup. The hardware consists of a virtual
# keithley, 2 virtual IVVI racks

# from functools import partial

import qtt.qtt_toymodel
reload(qtt.qtt_toymodel)
from qtt.qtt_toymodel import ModelError, DummyModel, VirtualIVVI, MockMeter, MockSource


model = DummyModel(name='dummymodel')
#l.setLevel(logging.DEBUG)
ivvi2 = VirtualIVVI(name='ivvi2', model=model)

#%%
ivvi1 = VirtualIVVI(name='ivvi1', model=model, gates=['c%d' % i for i in range(1, 17)])

keithley1 = MockMeter('keithley1', model=model)
keithley3 = MockMeter('keithley3', model=model)

source = MockSource('source', model=model)

# print('get P1: %f, %f, %f'  % (ivvi1.c1.get(), ivvi1.get('c1'), ivvi1.get_c1() ) )
# ivvi1.c1.set(300)
# print('get P1: %f, %f, %f'  % (ivvi1.c1.get(), ivvi1.get('c1'),
# ivvi1.get_c1() 

#%% We define virtual gates for the IVVI racks

reload(qtt.qtt_toymodel)
from qtt.qtt_toymodel import virtual_gates

gate_map = {
    # bias dacs
    'bias_1': (0, 1), 'bias_2': (0, 2),
    'bias_3': (1, 5), 'bias_4': (1, 6),

    # dacs creating the dot
    'P1': (0, 3), 'P1_fine': (0, 4),
    'P2': (0, 5), 'P2_fine': (0, 6),
    'P3': (1, 1), 'P3_fine': (1, 2),
    'P4': (1, 3),  'P4_fine': (1, 4),

    'L': (0, 8),
    'D2': (0, 9),
    'R': (0, 11),
    'T': (0, 13),
    'D1': (0, 15),
    'D3': (1, 9),

    # dacs creating the sensing dots
    'SD1a': (0, 14),  'SD1b': (1, 14), 'SD1c': (0, 16),
    'SD2a': (1, 7),   'SD2b': (1, 16), 'SD2c': (1, 12),
    'SD3a': (1, 13),  'SD3b': (1, 8),  'SD3c': (1, 15),
    'SD4a': (1, 11),  'SD4b': (0, 12), 'SD4c': (1, 10),
}

STOP

gates = virtual_gates(name='gates', gate_map=gate_map, instruments=[ivvi1, ivvi2])
self = gates
gate = 'P1'
# self.add_function('set_{}'.format(gate), call_cmd=partial(self._set,
# gate=gate), parameters=[Numbers()])

# f=partial(self._set, gate=gate)


for v in [-20, 0, 20, 40, 60]:
    gates.set_R(v)
    w = keithley3.readnext()
    print('v %f: w %f' % (v, w))


#%%
import qcodes.instrument_drivers.TimeStamp
ts = qcodes.instrument_drivers.TimeStamp.TimeStampInstrument(name='TimeStamp')


station = qc.Station(gates, source, keithley3, keithley1)
station.set_measurement(keithley3.amplitude, ts.timestamp)

#%%

dd = station.snapshot()
print(dd)

#%%

from qtt_toymodel import ParameterViewer

w = ParameterViewer(station)
w.setGeometry(1940,10,300,600)
self = w

x = self._itemsdict['gates']['R']

#%%

gates.set_T(101)
gates.set_R(np.random.rand())
gates.set_P1(np.random.rand())
w.updatedata()

#%%
station.snapshot()

#%%


# create custom viewer which gathers data from a station object
w = ParameterViewer(station)

w.updatecallback()

#%% Simple 1D scan loop


def scan1D(scanjob, station, location=None, delay=1.0):

    sweepdata = scanjob['sweepdata']
    param = getattr(gates, sweepdata['gate'])
    sweepvalues = param[sweepdata['start']:sweepdata['end']:sweepdata['step']]

    delay = scanjob.get('delay', delay)
    logging.warning('delay: %f' % delay)
    data = qc.Loop(sweepvalues, delay=delay).run(
        location=location, overwrite=True)

    return data


STOP

scanjob = dict(
    {'sweepdata': dict({'gate': 'R', 'start': -420, 'end': 220, 'step': .5}), 'delay': .01})
data = scan1D(scanjob, station, location='testsweep3')


data.sync()
data.arrays

#%%

plotQ = qc.QtPlot(data.amplitude)
plotQ.win.setGeometry(1920, 100, 800, 600)
plotQ.update()

#%%

data.sync()
data.arrays

#%%

qc.active_children()

qc.halt_bg()


#%%

# qc.active_children()[2].terminate()


#%%
plotQ.win.resize(700, 400)
g = plotQ.win.geometry

#%%

plotQ.win.setGeometry(1920, 100, 800, 600)

#%%


def timeProgress(data):
    ''' Simpe progress meter, should be integrated with either loop or data object '''
    data.sync()
    tt = data.arrays['timestamp']
    vv = ~np.isnan(tt)
    ttx = tt[vv]
    t0 = ttx[0]
    t1 = ttx[-1]

    logging.info('t0 %f t1 %f' % (t0, t1))

    fraction = ttx.size / tt.size[0]
    remaining = (t1 - t0) * (1 - fraction) / fraction
    return fraction, remaining


#%% Go!

for ii in range(1):
    print('progress: fraction %.2f, %.1f seconds remaining' %
          timeProgress(data))
    plotQ.update()
    time.sleep(.1)


#%%
scanjob = dict(
    {'sweepdata': dict({'gate': 'R', 'start': -420, 'end': 220, 'step': 2.5}), 'delay': .01})
data = scan1D(scanjob, station, location='testsweep4')
plotQ.add(data.amplitude)

#%%
plotQ.add(np.array(data.amplitude) + .2)


#%%

datax = qc.DataSet('testsweep2', mode=qcodes.DataMode.LOCAL)

fig = qc.MatPlot(datax.amplitude)

import pmatlab
pmatlab.tilefigs([fig.fig], [2, 2])


#%%


qcodes.DataSet.default_io = qcodes.DiskIO('/home/eendebakpt/tmp')

data = scan1D(scanjob, station, location=None)
print(data)
