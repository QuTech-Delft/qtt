import qtpy
#print(qtpy.API_NAME)

import numpy as np
import scipy
import os
import sys
import copy
import logging
import time
import qcodes
import qcodes as qc
import datetime

import qtpy.QtGui as QtGui
import qtpy.QtWidgets as QtWidgets

import matplotlib.pyplot as plt

from qtt.tools import tilefigs
import qtt.tools
from qtt.algorithms import analyseGateSweep
import qtt.live

from qtt.data import *

#%%

#%%

   
    
#%%

def plot1D(data, fig=100, mstyle='-b'):
    """ Show result of a 1D gate scan """
    
    
    kk=list(data.arrays.keys())
    
    if 'amplitude' in kk:
        val = 'amplitude'
    else:
        if 'readnext' in kk:
            val = 'readnext'
        else:
            val=kk[0]
        

    if fig is not None:
        plt.figure(fig)
        plt.clf()
        qc.MatPlot(getattr(data, val), num=fig)
        #plt.show()
        
if __name__=='__main__':
    plot1D(alldata, fig=100)
    
#%%
    
import time    
import pyqtgraph # FIXME
def complete(self, delay=1.0, txt=''):
        logging.info('waiting for data to complete')
        try:
            nloops=0
            while True:
                logging.info('%s waiting for data to complete (loop %d)' % (txt, nloops) )
                if self.sync()==False:
                    break
                time.sleep(delay)
                nloops=nloops+1
                try:
                    pyqtgraph.QtGui.QApplication.instance().processEvents()
                except:
                    print('error in processEvents...')                        
        except Exception as ex:
            return False
        return True
 
def getParams(station, keithleyidx):
    params=[]
    for x in keithleyidx:
        if isinstance(x, int):
            params+=[ getattr(station, 'keithley%d' % x).amplitude ]
        else:
            if isinstance(x, str):
                params+=[ getattr(station, x).amplitude ]
            else:
                params+=[ x ]
    return params    
    
def getDefaultParameter(data):
    if 'amplitude' in data.arrays.keys():
        return data.amplitude
    if 'amplitude_0' in data.arrays.keys():
        return data.amplitude_0
    if 'amplitude_1' in data.arrays.keys():
        return data.amplitude_1
    
    try:
        name = next(iter(data.arrays.keys()))
        return getattr(data, name)
    except:
        pass            
    return None
    
def scan1D(scanjob, station, location=None, delay=.01, liveplotwindow=None, background=False, title_comment=None):
    ''' Simple 1D scan '''
    gates=station.gates
    sweepdata = scanjob['sweepdata']
    gate=sweepdata.get('gate', None)
    if gate is None:
        gate=sweepdata.get('gates')[0]
    param = getattr(gates, gate)
    sweepvalues = param[sweepdata['start']:sweepdata['end']:sweepdata['step']]

    # legacy code...
    minstrument = scanjob.get('instrument', None)
    if minstrument is None:
        minstrument = scanjob.get('keithleyidx', None)
    params=getParams(station, minstrument)
    
    station.set_measurement(*params)

    delay = scanjob.get('delay', delay)
    logging.debug('delay: %f' % delay)
    print('scan1D: starting Loop (background %s)' % background)
    data = qc.Loop(sweepvalues, delay=delay, progress_interval=1).run(location=location, overwrite=True, background=background)
    data.sync()
    
    if liveplotwindow is None:        
        liveplotwindow = qtt.live.livePlot()        
    if liveplotwindow is not None:
        liveplotwindow.clear(); liveplotwindow.add( getDefaultParameter(data) )

    # FIXME
    if background:
        complete(data) #

    if not hasattr(data, 'metadata'):
        data.metadata=dict()
    sys.stdout.flush()

    return data

def scan2D(station, scanjob, title_comment='', liveplotwindow=None, wait_time=None, background=False):
    """ Make a 2D scan and create dictionary to store on disk """

    stepdata = scanjob['stepdata']
    sweepdata = scanjob['sweepdata']
    minstrument = scanjob.get('instrument', None)
    if minstrument is None:
        minstrument = scanjob.get('keithleyidx', None)
        
    print('fixme: compensategates')
    print('fixme: wait_time')
    #compensateGates = scanjob.get('compensateGates', [])
    #gate_values_corners = scanjob.get('gate_values_corners', [[]])

#    if wait_time == None:
#        wait_time = getwaittime(sweepdata['gates'][0])

    delay=scanjob.get('delay', 0.0)
        
    #readdevs = ['keithley%d' % x for x in keithleyidx]

    gates=station.gates
    
    sweepgate=sweepdata.get('gate', None)
    if sweepgate is None:
        sweepgate=sweepdata.get('gates')[0]

    stepgate=stepdata.get('gate', None)
    if stepgate is None:
        stepgate=stepdata.get('gates')[0]
    param = getattr(gates, sweepgate)
    stepparam = getattr(gates, stepgate)

    sweepvalues = param[sweepdata['start']:sweepdata['end']:sweepdata['step']]
    stepvalues = stepparam[stepdata['start']:stepdata['end']:stepdata['step']]
        
    logging.info('scan2D: %d %d'  % (len(stepvalues), len(sweepvalues)))
    logging.info('scan2D: delay %f'  % delay)
    innerloop = qc.Loop(stepvalues, delay=delay, progress_interval=2)
        


    t0=time.time()
    #alldata=innerloop.run(background=False)
    fullloop = innerloop.loop(sweepvalues, delay=delay)
    
    params=getParams(station, minstrument)

    measurement=fullloop.each( *params )
    
    alldata=measurement.run(background=background, data_manager=False)

    if liveplotwindow is None:        
        liveplotwindow = qtt.live.livePlot()        
    if liveplotwindow is not None:
        liveplotwindow.clear(); liveplotwindow.add( getDefaultParameter(alldata) )

    if background is True:
        alldata.complete()
    
    dt = time.time() - t0


    if not hasattr(alldata, 'metadata'):
        alldata.metadata=dict()
    alldata.metadata['scantime'] = str(datetime.datetime.now())
    alldata.metadata['scanjob'] = scanjob
        
    if 0:
        # FIXME...
        alldata = copy.copy(scanjob)
        alldata['scantime'] = str(datetime.datetime.now())
        alldata['wait_time'] = wait_time
        alldata['data_array'] = data.get_data()
        alldata['datadir'] = data._dir
        alldata['timemark'] = data._timemark
        alldata['allgatevalues'] = allgatevalues()
        alldata['gatevalues'] = gatevalues(activegates)
        alldata['gatevalues']['T'] = get_gate('T')
        alldata['dt'] = dt
        #basename='%s-sweep-2d-%s' % (idstr, 'x-%s-%s' % (gg[0], gg[2]) )
        #save(os.path.join(xdir, basename +'.pickle'), alldata )

    return alldata

#%% Measurement tools


def pinchoffFilename(g, od=None):
    ''' Return default filename of pinch-off scan '''
    if od is None:
        basename = 'pinchoff-sweep-1d-%s' % (g,)
    else:
        # old style filename
        basename = '%s-sweep-1d-%s' % (od['name'], g)
    return basename


def scanPinchValue(station, outputdir, gate, basevalues=None, keithleyidx=[1], cache=False, verbose=1, fig=10, full=0):
    basename = pinchoffFilename(gate, od=None)
    outputfile = os.path.join(outputdir, 'one_dot', basename + '.pickle')
    outputfile = os.path.join(outputdir, 'one_dot', basename)
    figfile = os.path.join(outputdir, 'one_dot', basename + '.png')

    if cache and os.path.exists(outputfile):
        print('  skipping pinch-off scans for gate %s' % (gate))
        print(outputfile)
        alldata = qcodes.load_data(outputfile)
        return alldata


    if basevalues is None:
        b = 0
    else:
        b = basevalues[gate]
    sweepdata = dict(
        {'gates': [gate], 'start': max(b, 0), 'end': -750, 'step': -2})
    if full == 0:
        sweepdata['step'] = -6

    scanjob = dict({'sweepdata': sweepdata, 'keithleyidx': keithleyidx, 'delay': 0.005})

    alldata = scan1D(scanjob, station, title_comment='scan gate %s' % gate)

    station.gates.set(gate, basevalues[gate])  # reset gate to base value

    # show results
    if fig is not None:
        plot1D(alldata, fig=fig)
        #plt.savefig(figfile)

    adata = analyseGateSweep(alldata, fig=None, minthr=None, maxthr=None)
    alldata.metadata['adata']=adata
    #  alldata['adata'] = adata
    
    writeDataset(outputfile, alldata)
    #alldata.write_to_disk(outputfile)
 #   pmatlab.save(outputfile, alldata)
    return alldata    

#%%

def getTimeString(t=None):
    """ Return time string for datetime.datetime object """
    if t == None:
        t = datetime.datetime.now()
    if type(t) == float:
        t = datetime.datetime.fromtimestamp(t)
    dstr = t.strftime('%H-%M-%S')
    return dstr
    
def getDateString(t=None, full=False):
    """ Return date string 

    Args:
        t : datetime.datetime
            time
    """
    if t is None:
        t = datetime.datetime.now()
    if type(t) == float:
        t = datetime.datetime.fromtimestamp(t)
    if full:
        dstr = t.strftime('%Y-%m-%d-%H-%M-%S')
    else:
        dstr = t.strftime('%Y-%m-%d')
    return dstr
    
def experimentFile(outputdir, tag=None, dstr=None, bname=None):
    """ Save experiment data for later analysis """
    if tag is None:
        tag = getDateString()
    if dstr is None:
        dstr = getDateString()

    basename = '%s' % (dstr,)
    if bname is not None:
        basename = '%s-' % bname + basename
    qtt.tools.mkdirc(os.path.join(outputdir, tag))
    pfile = os.path.join(outputdir, tag, basename + '.pickle')
    return pfile
    
#%%
def loadOneDotPinchvalues(od, outputdir, verbose=1):
    """ Load the pinch-off values for a one-dot 

    Arguments
    ---------
        od : dict
            one-dot structure
        outputdir : string
            location of the data
    
    """
    print('analyse data for 1-dot: %s' % od['name'] )
    gg=od['gates']
    pv=np.zeros( (3,1) )
    for ii,g in enumerate(gg):
        basename = pinchoffFilename(g, od=None)

        pfile=os.path.join(outputdir, 'one_dot', basename )
        alldata=loadDataset(pfile)
        #alldata,=pmatlab.load(pfile);  # alldata=alldata[0]
        if alldata is None:
            raise Exception('could not load file %s'  % pfile)
        adata=analyseGateSweep(alldata, fig=None, minthr=None, maxthr=None, verbose=1)
        if verbose:
            print('loadOneDotPinchvalues: pinchvalue for gate %s: %.1f'  % (g, adata['pinchvalue'] ) )
        pv[ii]=adata['pinchvalue']
    od['pinchvalues']=pv    
    return od
    

#%% Testing

if __name__=='__main__':    
    import qtt.scans
    reload(qtt.scans)
    od = qtt.scans.loadOneDotPinchvalues(od, outputdir, verbose=1)


#%%

    
if __name__=='__main__':    
    for gate in ['L', 'D1', 'D2', 'D3', 'R']+['P1','P2','P3','P4']: # ,'SD1a', 'SD1b', ''SD2a','SD]:
            alldata=scanPinchValue(station, outputdir, gate, basevalues=basevalues, keithleyidx=[3], cache=cache, full=full)

    