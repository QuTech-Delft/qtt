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

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s: %(message)s  (%(filename)s:%(lineno)d)', )

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

from qtt.qtt_toymodel import ParameterViewer

# create custom viewer which gathers data from a station object
w = ParameterViewer(station)
w.setGeometry(1940,10,300,600)
self = w

x = self._itemsdict['gates']['R']

#%%

gates.set_T(101)
gates.set_R(np.random.rand())
gates.set_P1(np.random.rand())
w.updatedata()

print('value: %f'  % keithley3.readnext() )

#%%
station.snapshot()

#%%

import inspect

def showCaller(offset=1):
    st=inspect.stack()
    print('function %s: caller: %s:%s name: %s' % (st[offset][3], st[offset+1][1], st[offset+1][2], st[offset+1][3] ) )


#w = ParameterViewer(station)

w.updatecallback()

#%% Simple 1D scan loop


def scan1D(scanjob, station, location=None, delay=1.0, qcodesplot=None, background=True):

    sweepdata = scanjob['sweepdata']
    param = getattr(gates, sweepdata['gate'])
    sweepvalues = param[sweepdata['start']:sweepdata['end']:sweepdata['step']]

    delay = scanjob.get('delay', delay)
    logging.debug('delay: %f' % delay)
    data = qc.Loop(sweepvalues, delay=delay).run(
        location=location, overwrite=True, background=background)

    if qcodesplot is not None:
        qcodesplot.clear(); qcodesplot.add(data.amplitude)

    sys.stdout.flush()

    return data


#%%
#data = qc.Loop(sweepvalues, delay=.001).run(location='dummy', overwrite=True, background=False, data_manager=False)

qtt.pythonVersion()
        
#%%

plotQ=None
#%%
scanjob = dict( {'sweepdata': dict({'gate': 'R', 'start': -160, 'end': 160, 'step': 2.}), 'delay': .01})
data = scan1D(scanjob, station, location='testsweep3', background=True)


data.sync()
data.arrays


#data = scan1D(scanjob, station, location='testsweep3', background=True)

#%

reload(qcodes); reload(qc); plotQ=None

#plotQ = qc.MatPlot(data.amplitude)
if plotQ is None:
    plotQ = qc.QtPlot(data.amplitude, windowTitle='Live plot', remote=False)
    plotQ.win.setGeometry(1920+360, 100, 800, 600)
    data.sync()    
    plotQ.update()
    w.callbacklist.append( plotQ.update )
else:
    data.sync()    
    plotQ.clear(); plotQ.add(data.amplitude)
    
#import qtpy
#from qtpy import QtCore, QtGui

#qtapp = QtGui.QApplication([])



#plotQ.clear(); plotQ.add(data.amplitude)
#data.sync(); data.arrays

STOP

#%%

#qc.active_children()
#qc.halt_bg()
#plotQ.win.setGeometry(1920, 100, 800, 600)


#%%


def timeProgress(data):
    ''' Simpe progress meter, should be integrated with either loop or data object '''
    data.sync()
    tt = data.arrays['timestamp']
    vv = ~np.isnan(tt)
    ttx = tt[vv]
    t0 = ttx[0]
    t1 = ttx[-1]

    logging.debug('t0 %f t1 %f' % (t0, t1))

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


qcodes.DataSet.default_io = qcodes.DiskIO('/home/eendebakpt/tmp/qdata')

scanjob = dict({'sweepdata': dict({'gate': 'R', 'start': -220, 'end': 220, 'step': 2.5}), 'delay': .01})
#scanjob = dict({'sweepdata': dict({'gate': 'R', 'start': 220, 'end': -220, 'step': -2.5}), 'delay': .01})

#%% Log file viewer

dd=os.listdir(qcodes.DataSet.default_io.base_location)


#%%

import qcodes
qcodes.DataSet.default_io = qcodes.DiskIO('/home/eendebakpt/tmp/qdata')

from qtt.logviewer import LogViewer

#l.setLevel(logging.DEBUG)


if __name__ == '__main__':
    logviewer = LogViewer()
    logviewer.setGeometry(1920+1280,60, 700,800)
    logviewer.qplot.win.setMaximumHeight(400)
    logviewer.show()
    self=logviewer
    
    
    logviewer.qplot.add(data.amplitude)



#%%
STOP


#%%
scanjob = dict({'sweepdata': dict({'gate': 'R', 'start': 220, 'end': -220, 'step': -2.5}), 'delay': .01})
data = scan1D(scanjob, station, location=None, qcodesplot=plotQ)
print(data)


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

#%%
#
# TODO: code refactoring
# TODO: merge back into qutech/packages (for now)
# TODO: an check or scan direction
# TODO: clean up code
import scipy
from pmatlab import cfigure
from pmatlab import *

def analyseGateSweep(dd, fig=None, minthr=None, maxthr=None, verbose=1, drawsmoothed=True, drawmidpoints=True):
    """ Analyse sweep of a gate for pinch value, low value and high value """

    goodgate = True
   
    if isinstance(data, qcodes.DataSet):
        XX=None
        g='R'
        value='amplitude'
        
        
        x=data.arrays[g]
        value=data.arrays[value]
        
        # detect direction of scan
        scandirection=np.sign(x[-1]-x[0])
        if scandirection<0 and 1:
            pass
            scandirection=1
            x=x[::-1]
            value=value[::-1]
    else:
        XX = dd['data_array']
        datashape = XX.shape
        goodgate = True
        sweepdata = dd['sweepdata']
        g = sweepdata['gates'][0]

        sr = np.arange(sweepdata['start'], sweepdata['end'], sweepdata['step'])
        if datashape[0] == 2 * sr.size:
                    # double sweep, reduce
            if verbose:
                print('analyseGateSweep: scan with sweepback: reducing data')
            XX = XX[0:sr.size, :]

        x = XX[:, 0]
        value = XX[:, 2]
        XX=None
        
    
    lowvalue = np.percentile(value, 1)
    highvalue = np.percentile(value, 90)
    # sometimes a channel is almost completely closed, then the percentile
    # approach does not function well
    ww = value[value > (lowvalue + highvalue) / 2]
    #[np.percentile(ww, 1), np.percentile(ww, 50), np.percentile(ww, 91) ]
    highvalue = np.percentile(ww, 90)

    if verbose >= 2:
        print('analyseGateSweep: lowvalue %.1f highvalue %.1f' %
              (lowvalue, highvalue))
    d = highvalue - lowvalue

    vv1 = value > (lowvalue + .2 * d)
    vv2 = value < (lowvalue + .8 * d)
    midpoint1 = vv1 * vv2
    ww = midpoint1.nonzero()[0]
    midpoint1 = np.mean(x[ww])

    # smooth signal
    kk = np.ones(3) / 3.
    ww = value
    for ii in range(4):
        ww = scipy.ndimage.filters.correlate1d(ww, kk, mode='nearest')
        #ww=scipy.signal.convolve(ww, kk, mode='same')
        #ww=scipy.signal.convolve2d(ww, kk, mode='same', boundary='symm')
    midvalue = .7 * lowvalue + .3 * highvalue
    if scandirection>=0:
        mp = (ww > (.7 * lowvalue + .3 * highvalue)).nonzero()[0][0]
    else:
        mp = (ww > (.7 * lowvalue + .3 * highvalue)).nonzero()[0][-1]
    midpoint2 = x[mp]
    
    if verbose >= 2:
        print('analyseGateSweep: midpoint2 %.1f midpoint1 %.1f' %
              (midpoint2, midpoint1))

    if minthr is not None and np.abs(lowvalue) > minthr:
        if verbose:
            print('analyseGateSweep: gate not good: gate is not closed')
            print(' minthr %s' % minthr)
        midpoint1 = np.percentile(x, .5)
        midpoint2 = np.percentile(x, .5)

        goodgate = False

    # check for gates that are fully open or closed
    if scandirection>0:    
        xleft = x[0:mp]
        leftval = ww[0:mp]
        rightval = ww[mp]
    else:
        xleft = x[mp:]
        leftval = ww[mp:]
        rightval = ww[0:mp]
    st = ww.std()
    if verbose:
        print('analyseGateSweep: leftval %.1f, rightval %.1f' %
              (leftval.mean(), rightval.mean()))
    if goodgate and (rightval.mean() - leftval.mean() < .3 * st):
        if verbose:
            print(
                'analyseGateSweep: gate not good: gate is not closed (or fully closed)')
        midpoint1 = np.percentile(x, .5)
        midpoint2 = np.percentile(x, .5)
        goodgate = False

    # fit a polynomial to the left side
    if goodgate and leftval.size > 5:
        # TODO: make this a robust fit
        fit = np.polyfit(xleft, leftval, 1)
        pp = np.polyval(fit, xleft)

        #pmid = np.polyval(fit, midpoint2)
        p0 = np.polyval(fit, xleft[-1])
        pmid = np.polyval(fit, xleft[0])
        if verbose:
	        print('p0 %.1f, pmid %.1f, leftval[0] %.1f' % (p0, pmid, leftval[0]))

        if pmid + (pmid - p0) * .25 > leftval[0]:
            midpoint1 = np.percentile(x, .5)
            midpoint2 = np.percentile(x, .5)
            goodgate = False
            if verbose:
                print(
                    'analyseGateSweep: gate not good: gate is not closed (or fully closed) (line fit check)')

    # another check on closed gates
    if scandirection>0:
        leftidx = range(0, mp)
        fitleft = np.polyfit(
            [x[leftidx[0]], x[leftidx[-1]]], [lowvalue, value[leftidx[-1]]], 1)
    else:
        leftidx = range(mp, value.shape[0])
        fitleft = np.polyfit(
            [x[leftidx[-1]], x[leftidx[0]]], [lowvalue, value[leftidx[0]]], 1)
    leftval = value[leftidx]

    #fitleft=np.polyfit([x[leftidx[-1]],x[leftidx[0]]], [lowvalue, midvalue], 1)

    leftpred = np.polyval(fitleft, x[leftidx])

    if np.abs( scandirection*(xleft[1]-xleft[0]) ) >150 and xleft.size>15:
        xleft0 = x[mp+6:]
        leftval0 = ww[mp+6:]
        fitL = np.polyfit(xleft0, leftval0, 1)
        pp = np.polyval(fitL, xleft0)
        nd=fitL[0]/(highvalue-lowvalue)
        if goodgate and (nd*750>1):
            midpoint1 = np.percentile(x, .5)
            midpoint2 = np.percentile(x, .5)
            goodgate = False
            if verbose:
                print('analyseGateSweep: gate not good: gate is not closed (or fully closed) (slope check)')
            pass        

    if np.sum(leftval - leftpred) > 0:
        midpoint1 = np.percentile(x, .5)
        midpoint2 = np.percentile(x, .5)
        goodgate = False
        if verbose:
            print(
                'analyseGateSweep: gate not good: gate is not closed (or fully closed) (left region check)')
        pass
    # gate not closed

    if fig is not None:
        cfigure(fig)
        plt.clf()
        plt.plot(x, value, '.-b', linewidth=2)
        plt.xlabel('Sweep %s [mV]' % g, fontsize=14)
        plt.ylabel('keithley [pA]', fontsize=14)

        if drawsmoothed:
            plt.plot(x, ww, '-g', linewidth=1)

        plot2Dline([0, -1, lowvalue], '--m', label='low value')
        plot2Dline([0, -1, highvalue], '--m', label='high value')

        if drawmidpoints:
            if verbose>=2:
                plot2Dline([-1, 0, midpoint1], '--g', linewidth=1)
            plot2Dline([-1, 0, midpoint2], '--m', linewidth=2)

        if verbose>=2: 
            #plt.plot(x[leftidx], leftval, '.r', markersize=15)
            plt.plot(x[leftidx], leftpred, '--r', markersize=15, linewidth=1)

        if verbose >= 2:
            1
           # plt.plot(XX[ww,0].astype(int), XX[ww,2], '.g')

    adata = dict({'pinchvalue': midpoint2 - 50,
                  'pinchvalueX': midpoint1 - 50, 'goodgate': goodgate})
    adata['lowvalue'] = lowvalue
    adata['highvalue'] = highvalue
    adata['xlabel'] = 'Sweep %s [mV]' % g
    adata['mp'] = mp
    adata['midpoint'] = midpoint2
    adata['midvalue'] = midvalue

    if verbose >= 2:
        print('analyseGateSweep: gate status %d: pinchvalue %.1f' %
              (goodgate, adata['pinchvalue']))
        adata['Xsmooth'] = ww
        adata['XX'] = XX
        adata['X'] = value
        adata['x'] = x
        adata['ww'] = ww

    return adata
    

dd=data
adata=analyseGateSweep(dd, fig=10, verbose=2)

