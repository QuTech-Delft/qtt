""" Basic scan functions

This module contains functions for basic scans, e.g. scan1D, scan2D, etc.
This is part of qtt. 
"""

import time
import numpy as np
import scipy
import os
import sys
import copy
import logging
import time
import datetime
import warnings
import pyqtgraph as pg
import skimage
import skimage.filters
import matplotlib.pyplot as plt

import qcodes
import qcodes as qc
from qcodes.utils.helpers import tprint
from qcodes.instrument.parameter import Parameter, StandardParameter, ManualParameter
from qcodes import DataArray
from qcodes.plots.qcmatplotlib import MatPlot
from qcodes import Instrument

import qtt.tools
from qtt.tools import tilefigs
from qtt.algorithms.gatesweep import analyseGateSweep
import qtt.algorithms.onedot  
import qtt.live
from qtt.tools import deprecated

from qtt.data import makeDataSet1D, makeDataSet2D, makeDataSet1Dplain
from qtt.data import diffDataset, experimentFile, loadDataset, writeDataset
from qtt.data import uniqueArrayName

from qtt.tools import update_dictionary
from qtt.structures import VectorParameter

#%%


def checkReversal(im0, verbose=0):
    """ Check sign of a current scan

    We assume that the current is either zero or positive 
    Needed when the keithley (or some other measurement device) has been reversed
    """
    thr = skimage.filters.threshold_otsu(im0)
    mval = np.mean(im0)

    # meanopen = np.mean(im0[:,:])
    fr = thr < mval
    #fr = thr < 0
    if verbose:
        print(' checkReversal: %d (mval %.1f, thr %.1f)' % (fr, mval, thr))
    if fr:
        return -1
    else:
        return 1


def fixReversal(im0, verbose=0):
    """ Fix sign of a current scan

    We assume that the current is either zero or positive 
    Needed when the keithley (or some other measurement device) has been reversed
    """
    r = checkReversal(im0, verbose=verbose)
    return r * np.array(im0)

#%%


def instrumentName(namebase):
    """ Return name for qcodes instrument that is available
    
    Args:
        namebase (str)
    Returns:
        name (str)
    """
    inames=qcodes.Instrument._all_instruments
    name=namebase
    for ii in range(10000):
        if not( name in inames):
            return name
        else:
             name = namebase+'%d' % ii   
    raise Exception('could not find unique name for instrument with base %s' % namebase)
    
def createScanJob(g1, r1, g2=None, r2=None, step=-1, keithleyidx='keithley1'):
    """ Create a scan job

    Arguments
    ---------
    g1 (str): sweep gate
    r1 (array, list): Range to sweep
    g2 (str, optional): step gate
    r2 (array, list): Range to step
    step (int, optional): Step value (default is -1)

    """
    stepdata = scanjob_t(
        {'param': [g1], 'start': r1[0], 'end': r1[1], 'step': step})
    scanjob = dict({'sweepdata': sweepdata, 'minstrument': keithleyidx})
    if not g2 is None:
        stepdata = dict(
            {'param': [g2], 'start': r2[0], 'end': r2[1], 'step': step})
        scanjob['stepdata'] = stepdata

    return scanjob

#%%


def parse_stepdata(stepdata):
    """ Helper function for legacy code """
    if not isinstance(stepdata, dict):
        raise Exception('stepdata should be dict structure')

    v = stepdata.get('gates', None)
    if v is not None:
        raise Exception('please use param instead of gates')
    v = stepdata.get('gate', None)
    if v is not None:
        warnings.warn('please use param instead of gates', DeprecationWarning)
        stepdata['param'] = stepdata['gate']

    v = stepdata.get('param', None)
    if isinstance(v, (str, StandardParameter, ManualParameter, dict)):
        pass
    elif isinstance(v, list):
        warnings.warn('please use string or Instrument instead of list')
        stepdata['param'] = stepdata['param'][0]

    if 'range' in stepdata:
        if 'end' in 'stepdata':
            if stepdata['end']!=stepdata['start']+stepdata['range']:
                warnings.warn('in scanjob the start, end and range arguments do not match')
        stepdata['end']=stepdata['start']+stepdata['range']
    return stepdata


def get_param(gates, sweepgate):
    """ Get qcodes parameter from scanjob argument """
    if isinstance(sweepgate, str):
        return getattr(gates, sweepgate)
    else:
        # assume the argument already is a parameter
        return sweepgate

def get_param_name(gates, sweepgate):
    """ Get qcodes parameter name from scanjob argument """
    if isinstance(sweepgate, str):
        return sweepgate
    else:
        # assume the argument already is a parameter
        return sweepgate.name


from qtt.algorithms.generic import findCoulombDirection
from qtt.data import dataset2Dmetadata, dataset2image


def onedotHiresScan(station, od, dv=70, verbose=1, fig=4000, ptv=None):
    """ Make high-resolution scan of a one-dot """
    if verbose:
        print('onedotHiresScan: one-dot: %s' % od['name'])

    # od, ptv, pt,ims,lv, wwarea=onedotGetBalance(od, alldata, verbose=1, fig=None)
    if ptv is None:
        ptv = od['balancepoint']
    keithleyidx = [od['instrument']]
    scanjobhi = createScanJob(od['gates'][0], [float(ptv[1]) + 1.2 * dv, float(ptv[1]) - 1.2 * dv], g2=od[
                              'gates'][2], r2=[float(ptv[0]) + 1.2 * dv, float(ptv[0]) - 1.2 * dv], step=-4)
    scanjobhi['minstrument'] = keithleyidx
    scanjobhi['stepdata']['end'] = max(scanjobhi['stepdata']['end'], -780)
    scanjobhi['sweepdata']['end'] = max(scanjobhi['sweepdata']['end'], -780)

    wait_time = waitTime(od['gates'][2], station=station)
    scanjobhi['sweepdata']['wait_time'] = wait_time
    scanjobhi['stepdata']['wait_time'] = 2*waitTime(None, station) + 3 * wait_time

    alldatahi = qtt.measurements.scans.scan2D(station, scanjobhi)
    extentscan, g0, g2, vstep, vsweep, arrayname = dataset2Dmetadata(
        alldatahi, verbose=0, arrayname=None)
    impixel, tr = dataset2image(alldatahi, mode='pixel')

    ptv, fimg, tmp = qtt.algorithms.onedot.onedotGetBalanceFine(
        impixel, alldatahi, verbose=1, fig=fig)

    if tmp['accuracy'] < .2:
        logging.info('use old data point!')
        # use normal balance point (fixme)
        ptv = od['balancepoint']
        ptx = od['balancepointpixel'].reshape(1, 2)
    else:
        ptx = tmp['ptpixel'].copy()
    step = scanjobhi['stepdata']['step']
    val = findCoulombDirection(
        impixel, ptx, step, widthmv=8, fig=None, verbose=1)
    od['coulombdirection'] = val

    od['balancepointfine'] = ptv
    od['setpoint'] = ptv + 10

    alldatahi.metadata['od'] = od

    scandata = dict({'od': od, 'dataset': alldatahi, 'scanjob': scanjobhi})
    return scandata, od


#%%


def plot1D(data, fig=100, mstyle='-b'):
    """ Show result of a 1D gate scan """

    val = data.default_parameter_name()

    if fig is not None:
        plt.figure(fig)
        plt.clf()
        MatPlot(getattr(data, val), interval=None, num=fig)

#%%


def get_instrument(instr, station=None):
    """ Return handle to instrument
    
    Args:
        instr (str, Instrument ): name of instrument or handle
    """
    
    if isinstance(instr, Instrument):
        return instr
    
    if not isinstance(instr, str):
        raise Exception('could not find instrument %s' % str(instr))
    try:
        ref = Instrument.find_instrument(instr)
        return ref
    except:
        pass
    if station is not None:
        if instr in station.components:
            ref=station.conponents[instr]
            return ref
    raise Exception('could not find instrument %s' % str(instr))

def get_measurement_params(station, mparams):
    """ Get qcodes parameters from an index or string or parameter """
    params = []
    if isinstance(mparams, (int, str, Parameter)):
        # for convenience
        mparams = [mparams]
    elif isinstance(mparams, (list, tuple)):
        pass
    else:
        warnings.warn('unknown argument type')
    for x in mparams:
        if isinstance(x, int):
            params += [getattr(station, 'keithley%d' % x).amplitude]
        elif isinstance(x, str):
            if x.startswith('digitizer'):
                params += [getattr(station.digitizer, 'channel_%c' % x[-1])]
            else:
                params += [getattr(station, x).amplitude]
        else:
            params += [x]
    return params


def getDefaultParameter(data):
    """ Return name of the main array in the dataset """
    return data.default_parameter_name()

#%%
def scan1D(station, scanjob, location=None, liveplotwindow=None, plotparam='measured', verbose=1):
    """Simple 1D scan. 

    Args:
        station (object): contains all data on the measurement station
        scanjob (scanjob_t): data for scan

    Returns:
        alldata (DataSet): contains the measurement data and metadata
    """
    gates = station.gates
    gatevals = gates.allvalues()

    minstrument = scanjob.get('minstrument', None)
    mparams = get_measurement_params(station, minstrument)

    if type(scanjob) is dict:
        warnings.warn('Use the scanjob_t class.', DeprecationWarning)
        scanjob = scanjob_t(scanjob)

    scanjob.parse_stepdata('sweepdata')

    if isinstance(scanjob['sweepdata']['param'], lin_comb_type):
        scanjob['scantype'] = 'scan1Dvec'
        scanjob._start_end_to_range()
    else:
        scanjob['scantype'] = 'scan1D'

    sweepdata = scanjob['sweepdata']

    sweepvalues = scanjob._convert_scanjob_vec(station)

    wait_time = sweepdata.get('wait_time', 0)
    wait_time_startscan = scanjob.get('wait_time_startscan', 0)
    t0 = time.time()

    # LEGACY
    instrument = scanjob.get('instrument', None)
    if instrument is not None:
        raise Exception('legacy argument instrument: use minstrument instead!')

    logging.debug('wait_time: %s' % str(wait_time))

    alldata, (set_names, measure_names) = makeDataSet1D(sweepvalues, yname=mparams, location=location, loc_record={'label': scanjob['scantype']}, return_names=True)

    if liveplotwindow is None:
        liveplotwindow = qtt.live.livePlot()
    if liveplotwindow:
        liveplotwindow.clear()
        liveplotwindow.add(alldata.default_parameter_array(paramname=plotparam))

    def myupdate():
        if liveplotwindow:
            t0 = time.time()            
            liveplotwindow.update()
            if verbose >= 2:
                print('scan1D: myupdate: %.3f ' % (time.time() - t0))

    tprev=time.time()
    for ix, x in enumerate(sweepvalues):
        if verbose:
            tprint('scan1D: %d/%d: time %.1f' % (ix, len(sweepvalues), time.time() - t0), dt=1.5)

        if scanjob['scantype'] == 'scan1Dfastvec':
            for param in scanjob['phys_gates_vals']:
                gates.set(param, scanjob['phys_gates_vals'][param][ix])
        else:
            sweepvalues.set(x)
        if ix == 0:
            qtt.time.sleep(wait_time_startscan)
        else:
            time.sleep(wait_time)
        for ii, p in enumerate(mparams):
            value = p.get()
            alldata.arrays[measure_names[ii]].ndarray[ix] = value
      
        
        delta, tprev, update_plot = delta_time(tprev, thr=.5)
        if (liveplotwindow) and update_plot:
            myupdate()

    myupdate()
    dt = time.time() - t0

    if scanjob['scantype'] is 'scan1Dvec':
        for param in scanjob['phys_gates_vals']:
            parameter = gates.parameters[param]
            arr = DataArray(name=parameter.name, array_id=parameter.name, label=parameter.label, unit=parameter.unit, preset_data=scanjob['phys_gates_vals'][param], set_arrays=(alldata.arrays[sweepvalues.parameter.name],))
            alldata.add_array(arr)

    if not hasattr(alldata, 'metadata'):
        alldata.metadata = dict()

    update_dictionary(alldata.metadata, scanjob=scanjob, dt=dt, station=station.snapshot())
    update_dictionary(alldata.metadata, scantime=str(datetime.datetime.now()), allgatevalues=gatevals)

    logging.info('scan1D: done %s' % (str(alldata.location),))

    alldata.write(write_metadata=True)

    return alldata


#%%
def scan1Dfast(station, scanjob, location=None, liveplotwindow=None, verbose=1):
    """Fast 1D scan. 

    Args:
        station (object): contains all data on the measurement station
        scanjob (scanjob_t): data for scan

    Returns:
        alldata (DataSet): contains the measurement data and metadata
    """
    gates = station.gates
    gatevals = gates.allvalues()

    if 'sd' in scanjob:
        warnings.warn('sd argument is not supported in scan1Dfast')

    if type(scanjob) is dict:
        warnings.warn('Use the scanjob_t class.', DeprecationWarning)
        scanjob = scanjob_t(scanjob)

    scanjob.parse_stepdata('sweepdata')

    minstrhandle = get_instrument(scanjob.get('minstrumenthandle', 'fpga'), station=station)

    read_ch = scanjob['minstrument']
    if isinstance(read_ch, int):
        read_ch = [read_ch]

    if isinstance(scanjob['sweepdata']['param'], lin_comb_type):
        scanjob['scantype'] = 'scan1Dfastvec'
        scanjob._start_end_to_range()
    else:
        scanjob['scantype'] = 'scan1Dfast'

    sweepdata = scanjob['sweepdata']

    Naverage = scanjob.get('Naverage', 20)

    period = scanjob['sweepdata'].get('period', 1e-3)

    t0 = qtt.time.time()
    
    wait_time_startscan = scanjob.get('wait_time_startscan', 0)

    if scanjob['scantype'] == 'scan1Dfast':
        sweeprange = (sweepdata['end'] - sweepdata['start'])
        waveform, sweep_info = station.awg.sweep_gate(sweepdata['param'], sweeprange, period)
        sweepgate_value = (sweepdata['start'] + sweepdata['end']) / 2
        gates.set(sweepdata['param'], float(sweepgate_value))
    else:
        sweeprange = sweepdata['range']
        waveform, sweep_info = station.awg.sweep_gate_virt(sweepdata['param'], sweeprange, period)

    qtt.time.sleep(wait_time_startscan)

    data = measuresegment(waveform, Naverage, station, minstrhandle, read_ch,
                          period=period, sawtooth_width=waveform['width' ])

    sweepvalues = scanjob._convert_scanjob_vec(station, data.size)

    alldata = makeDataSet1Dplain(sweepvalues.parameter.name, sweepvalues, ['measured%d' % i for i in read_ch], data, location=location, loc_record={'label': scanjob['scantype']})
    
    station.awg.stop()

    if liveplotwindow is None:
        liveplotwindow = qtt.live.livePlot()
    if liveplotwindow is not None:
        liveplotwindow.clear()
        liveplotwindow.add(alldata.default_parameter_array())

    dt = time.time() - t0

    if scanjob['scantype'] is 'scan1Dfastvec':
        for param in scanjob['phys_gates_vals']:
            parameter = gates.parameters[param]
            arr = DataArray(name=parameter.name, array_id=parameter.name, label=parameter.label, unit=parameter.unit, preset_data=scanjob['phys_gates_vals'][param], set_arrays=(alldata.arrays[sweepvalues.parameter.name],))
            alldata.add_array(arr)

    if not hasattr(alldata, 'metadata'):
        alldata.metadata = dict()

    update_dictionary(alldata.metadata, scanjob=scanjob, dt=dt, station=station.snapshot())
    update_dictionary(alldata.metadata, scantime=str(datetime.datetime.now()), allgatevalues=gatevals)

    alldata = qtt.tools.stripDataset(alldata)

    alldata.write(write_metadata=True)

    return alldata

#%%


def wait_bg_finish(verbose=0):
    """ Wait for background job to finish """
    if not hasattr(qcodes, 'get_bg'):
        return True
    for ii in range(10):
        m = qcodes.get_bg()
        if verbose:
            print('wait_bg_finish: loop %d: bg %s ' % (ii, m))
        if m is None:
            break
        time.sleep(0.05)
    m = qcodes.get_bg()
    if verbose:
        print('wait_bg_finish: final: bg %s ' % (m, ))
    if m is not None:
        logging.info('background job not finished')
    return m is None


def makeScanjob(sweepgates, values, sweepranges, resolution):
    """ Create a scanjob from sweep ranges and a centre """
    sj = {}

    nx = len(sweepgates)
    step = sweepranges[0] / resolution[0]
    stepdata = {'gates': [sweepgates[0]], 'start': values[0] - sweepranges[0] / 2, 'end': values[0] + sweepranges[0] / 2, 'step': step}
    sj['stepdata'] = stepdata
    if nx == 2:
        step = sweepranges[1] / resolution[1]
        sweepdata = {'gates': [sweepgates[1]], 'start': values[1] - sweepranges[1] / 2, 'end': values[1] + sweepranges[0] / 2, 'step': step}
        sj['sweepdata'] = sweepdata
        sj['wait_time_step'] = 4
    return sj

#%%

class scanjob_t(dict):
    """ Structure that contains information about a scan 
    
    A typical scanjob contains the following fields:
        
    Fields:
        sweepdata (dict):
        stepdata (dict)
        minstrument (str, Parameter or tuple)
    
    The sweepdata and stepdata are structures with the following fields:
        
        param (str, Parameter or dict): parameter to vary
        start, end, step (float)
        wait_time (float)
        
    If the param field 
    
    Note: currently the scanjob_t is a thin wrapper around a dict.
    """

    def parse_stepdata(self, field):
        """ Helper function for legacy code """
        stepdata = self[field]
        if not isinstance(stepdata, dict):
            raise Exception('%s should be dict structure' % field)
    
        v = stepdata.get('gates', None)
        if v is not None:
            raise Exception('please use param instead of gates')
        v = stepdata.get('gate', None)
        if v is not None:
            warnings.warn('please use param instead of gates', DeprecationWarning)
            stepdata['param'] = stepdata['gate']
    
        v = stepdata.get('param', None)
        if isinstance(v, (str, StandardParameter, ManualParameter, dict)):
            pass
        elif isinstance(v, list):
            warnings.warn('please use string or Instrument instead of list')
            stepdata['param'] = stepdata['param'][0]
        self[field] = stepdata

    def _start_end_to_range(self):
        """ Add range to stepdata and/or sweepdata in scanjob. """

        scanfields = ['stepdata', 'sweepdata']

        for scanfield in scanfields:
            if scanfield in self:
                scaninfo = self[scanfield]
                if 'range' not in scaninfo:
                    scaninfo['range'] = scaninfo['end'] - scaninfo['start']
                    warnings.warn('Start and end are converted to a range to scan around the current dc values.')
                    scaninfo['start'] = -scaninfo['range']/2
                    scaninfo['end'] = scaninfo['range']/2
                else:
                    pass

    def _parse_2Dvec(self):
        """ Adjust the parameter field in the step- and sweepdata for 2D vector scans.
        
        This adds coefficient zero for parameters in either the sweep- 
        or the step-parameters that do not exist in the other.
        
        """
        stepdata = self['stepdata']
        sweepdata = self['sweepdata']
        params = set()
        vec_check = [(stepdata, isinstance(stepdata['param'], lin_comb_type)), (sweepdata, isinstance(sweepdata['param'], lin_comb_type))]
        for scaninfo, boolean in vec_check:
            if boolean is False:
                scaninfo['param'] = {scaninfo['param']: 1}
        params.update(list(stepdata['param'].keys()))
        params.update(list(sweepdata['param'].keys()))
        for param in params:
            if param not in stepdata['param']:
                stepdata['param'][param] = 0
            if param not in sweepdata['param']:
                sweepdata['param'][param] = 0
        self['stepdata'] = stepdata
        self['sweepdata'] = sweepdata

    def _convert_scanjob_vec(self, station, sweeplength=None, steplength=None):
        """ Adjust the scanjob for vector scans. 

        Note: For vector scans the range field in stepdata and sweepdata is 
        used by default. If only start and end are given a range will be 
        calculated from those, but only the difference between them is used for
        vector scans.

        Args:
            station (object): contains all the instruments

        Returns:
            scanjob (scanjob_t): updated data for scan
            scanvalues (list): contains the values for parameters to scan over
        """
        gates = station.gates
        
        if self['scantype'][:6] == 'scan1D':
            sweepdata = self['sweepdata']
            if self['scantype'] in ['scan1Dvec', 'scan1Dfastvec']:
                sweepname = 'sweepparam'
                sweepparam = VectorParameter(name=sweepname, comb_map=[(gates.parameters[x], sweepdata['param'][x]) for x in sweepdata['param']])
            elif self['scantype'] in ['scan1D', 'scan1Dfast']:
                sweepparam = get_param(gates, sweepdata['param'])
            else:
                raise Exception('unknown scantype')
            if sweeplength is not None:
                sweepdata['step'] = (sweepdata['end'] - sweepdata['start']) / sweeplength
            if self['scantype'] in ['scan1Dvec', 'scan1Dfastvec']:
                last=sweepdata['start']+sweepdata['range']
                scanvalues = sweepparam[sweepdata['start']:last:sweepdata['step']]

                param_init = {param: gates.get(param) for param in sweepdata['param']}
                self['phys_gates_vals'] = {param: np.zeros(len(scanvalues)) for param in sweepdata['param']}
                sweep_array = np.arange(-sweepdata['range']/2, sweepdata['range']/2, sweepdata['step'])  
                for param in sweepdata['param']:
                    self['phys_gates_vals'][param] = param_init[param] + sweep_array * sweepdata['param'][param]
            else:
                scanvalues = sweepparam[sweepdata['start']:sweepdata['end']:sweepdata['step']]

            self['sweepdata'] = sweepdata
        elif self['scantype'][:6] == 'scan2D':
            stepdata = self['stepdata']
            sweepdata = self['sweepdata']
            if self['scantype'] in ['scan2Dvec', 'scan2Dfastvec', 'scan2Dturbovec']:
                stepname = 'stepparam'
                sweepname = 'sweepparam'
                if not (np.dot(list(stepdata['param'].values()), [sweepdata['param'][x] for x in stepdata['param']]) == 0):
                    stepname = stepname + '_v'
                    sweepname= sweepname + '_v'
                stepparam = VectorParameter(name=stepname, comb_map=[(gates.parameters[x], stepdata['param'][x]) for x in stepdata['param']])
                sweepparam = VectorParameter(name=sweepname, comb_map=[(gates.parameters[x], sweepdata['param'][x]) for x in sweepdata['param']])
            elif self['scantype'] in ['scan2D', 'scan2Dfast', 'scan2Dturbo']:
                stepgate = stepdata.get('param', None)
                stepparam = get_param(gates, stepgate)
                sweepgate = sweepdata.get('param', None)
                sweepparam = get_param(gates, sweepgate)
            else:
                raise Exception('unknown scantype')
            if sweeplength is not None:
                sweepdata['step'] = (sweepdata['end'] - sweepdata['start']) / sweeplength
            if steplength is not None:
                stepdata['step'] = (stepdata['end'] - stepdata['start']) / steplength
                        
            sweepvalues = sweepparam[sweepdata['start']:sweepdata['end']:sweepdata['step']]
            stepvalues = stepparam[stepdata['start']:stepdata['end']:stepdata['step']]
            scanvalues = [stepvalues, sweepvalues]
            if self['scantype'] in ['scan2Dvec', 'scan2Dfastvec', 'scan2Dturbovec']:
                param_init = {param: gates.get(param) for param in sweepdata['param']}
                self['phys_gates_vals'] = {param: np.zeros((len(stepvalues), len(sweepvalues))) for param in sweepdata['param']}
                step_array2d = np.tile(np.arange(-stepdata['range']/2, stepdata['range']/2, stepdata['step']).reshape(-1, 1), (1, len(sweepvalues)))
                sweep_array2d = np.tile(np.arange(-sweepdata['range']/2, sweepdata['range']/2, sweepdata['step']), (len(stepvalues), 1))   
                for param in sweepdata['param']:
                    self['phys_gates_vals'][param] = param_init[param] + step_array2d * stepdata['param'][param] + sweep_array2d * sweepdata['param'][param]    
            self['stepdata'] = stepdata
            self['sweepdata'] = sweepdata

        return scanvalues

def delta_time(tprev, thr=2):
    """ Helper function to prevent too many updates """
    t = time.time()
    update = 0
    delta = t - tprev
    if delta > thr:
        tprev = t
        update = 1
    return delta, tprev, update


def parse_minstrument(scanjob):
    """ Extract the parameters to be measured from the scanjob """
    minstrument = scanjob.get('minstrument', None)
    if minstrument is None:
        warnings.warn('use minstrument instead of keithleyidx')
        minstrument = scanjob.get('keithleyidx', None)

    return minstrument


lin_comb_type = dict
""" Class to represent linear combinations of parameters  """

def scan2D(station, scanjob, location=None, liveplotwindow=None, plotparam='measured', diff_dir=None, verbose=1):
    """Make a 2D scan and create dictionary to store on disk.

    For 2D vector scans see also the documentation of the _convert_scanjob_vec
    method of the scanjob_t class.

    Args:
        station (object): contains all the instruments
        scanjob (scanjob_t): data for scan

    Returns:
        alldata (DataSet): contains the measurement data and metadata
    """
    gates = station.gates
    gatevals = gates.allvalues()

    minstrument = parse_minstrument(scanjob)
    mparams = get_measurement_params(station, minstrument)

    if type(scanjob) is dict:
        warnings.warn('Use the scanjob_t class.', DeprecationWarning)
        scanjob = scanjob_t(scanjob)

    stepdata = parse_stepdata(scanjob['stepdata'])
    sweepdata = parse_stepdata(scanjob['sweepdata'])

    if isinstance(scanjob['stepdata']['param'], lin_comb_type) or isinstance(scanjob['sweepdata']['param'], lin_comb_type):
        scanjob['scantype'] = 'scan2Dvec'
        scanjob._start_end_to_range()
        scanjob._parse_2Dvec()
    else:
        scanjob['scantype'] = 'scan2D'

    scanvalues = scanjob._convert_scanjob_vec(station)
    stepvalues = scanvalues[0]
    sweepvalues = scanvalues[1]

    stepdata = scanjob['stepdata']
    sweepdata = scanjob['sweepdata']

    wait_time_sweep = sweepdata.get('wait_time', 0)
    wait_time_step = stepdata.get('wait_time', 0)
    wait_time_startscan = scanjob.get('wait_time_startscan', 0)
    logging.info('scan2D: %d %d' % (len(stepvalues), len(sweepvalues)))
    logging.info('scan2D: wait_time_sweep %f' % wait_time_sweep)
    logging.info('scan2D: wait_time_step %f' % wait_time_step)

    alldata, (set_names, measure_names) = makeDataSet2D(stepvalues, sweepvalues,
                                                        measure_names=mparams, location=location, loc_record={'label': scanjob['scantype']},
                                                        return_names=True)

    if verbose >= 2:
        print('scan2D: created dataset')
        print('  set_names: %s ' % (set_names,))
        print('  measure_names: %s ' % (measure_names,))

    t0 = qtt.time.time()

    if liveplotwindow is None:
        liveplotwindow = qtt.live.livePlot()
    if liveplotwindow:
        liveplotwindow.clear()
        liveplotwindow.add(alldata.default_parameter_array(paramname=plotparam))

    tprev = time.time()
    for ix, x in enumerate(stepvalues):
        if verbose:
            tprint('scan2D: %d/%d: time %.1f: setting %s to %.3f' % (ix, len(stepvalues), time.time() - t0, stepvalues.name, x), dt=1.5)
        if scanjob['scantype'] == 'scan2Dvec':
            pass
        else:
            stepvalues.set(x)
        for iy, y in enumerate(sweepvalues):
            if scanjob['scantype'] == 'scan2Dvec':
                for param in scanjob['phys_gates_vals']:
                    gates.set(param, scanjob['phys_gates_vals'][param][ix, iy])
            else:
                sweepvalues.set(y)
            if iy == 0:
                if ix == 0:
                    qtt.time.sleep(wait_time_startscan)
                else:
                    qtt.time.sleep(wait_time_step)
            else:
                time.sleep(wait_time_sweep)

            for ii, p in enumerate(mparams):
                value = p.get()
                alldata.arrays[measure_names[ii]].ndarray[ix, iy] = value

        if ix == len(stepvalues) - 1 or ix % 5 == 0:
            delta, tprev, update = delta_time(tprev, thr=.2)
            if update and liveplotwindow:
                liveplotwindow.update_plot()
                pg.mkQApp().processEvents()

        if qtt.abort_measurements():
            print('  aborting measurement loop')
            break
    dt = qtt.time.time() - t0

    if liveplotwindow:
        liveplotwindow.update_plot()

    if diff_dir is not None:
        alldata = diffDataset(alldata, diff_dir=diff_dir, fig=None)

    if scanjob['scantype'] == 'scan2Dvec':
        for param in scanjob['phys_gates_vals']:
            parameter = gates.parameters[param]
            arr = DataArray(name=parameter.name, array_id=parameter.name, label=parameter.label, unit=parameter.unit, preset_data=scanjob['phys_gates_vals'][param], set_arrays=(alldata.arrays[stepvalues.parameter.name], alldata.arrays[sweepvalues.parameter.name]))
            alldata.add_array(arr)

    if not hasattr(alldata, 'metadata'):
        alldata.metadata = dict()

    update_dictionary(alldata.metadata, scanjob=scanjob, dt=dt, station=station.snapshot())
    update_dictionary(alldata.metadata, scantime=str(datetime.datetime.now()), allgatevalues=gatevals)

    alldata.write(write_metadata=True)

    return alldata


#%%

def process_digitizer_trace(data, width, period, samplerate, padding=0,
                            fig=None, pre_trigger=None):
    """ Process data from the M4i and a sawtooth trace 
    
    This is done to remove the extra padded data of the digitized and to 
    extract the forward trace of the sawtooth.
    
    Args:
        data (Nxk array)
        width (float): width of the sawtooth
        period (float)
        samplerate (float)
    Returns
        processed_data (Nxk array): processed data
        rr (tuple)
    """
    
    npoints = period    *samplerate # expected number of points
    rwidth=1-width
    if pre_trigger is None:
        # assume trigger is in middle of trace
        cctrigger=data.shape[0]/2 # position of trigger in signal
    else:
        cctrigger=pre_trigger # position of trigger in signal
    
    
    cc = data.shape[1]/2 # centre of sawtooth
    cc = cctrigger + width*npoints/2+0
    npoints2=width*npoints
    npoints2=npoints2-(npoints2%2)
    r1=int(cc-npoints2/2)-padding
    r2=int(cc+npoints2/2)+padding
    processed_data=data[ r1:r2,:]
    if fig is not None:
        plt.figure(fig); plt.clf();
        plt.plot(data, label='raw data' )
        plt.title('Processing of digitizer trace' )
        plt.axis('tight')
        #cc=data.shape[0]*(.5-rwidth/2)
        #dcc=int(data.shape[0]/2)
        #cc=dcc
        
        qtt.pgeometry.plot2Dline([-1,0,cctrigger], ':g', linewidth=3, label='trigger' )        
        qtt.pgeometry.plot2Dline([-1,0,cc], '-c', linewidth=1, label='centre of sawtooth', zorder=-10 )        
        qtt.pgeometry.plot2Dline([0,-1,0,], '-', color=(0,1,0,.41),linewidth=.8 )
        
        qtt.pgeometry.plot2Dline([-1,0,r1], ':k', label='range of forward slope')
        qtt.pgeometry.plot2Dline([-1,0,r2], ':k')
    
        qtt.pgeometry.plot2Dline([-1,0,cc+samplerate*period*(width/2+rwidth)], '--m', label='?')
        qtt.pgeometry.plot2Dline([-1,0,cc+samplerate*period*-(width/2+rwidth) ], '--m')
        plt.legend(numpoints=1)
    return processed_data, (r1, r2)

def select_digitizer_memsize(digitizer, period, verbose=1, pre_trigger=None):
    """ Select suitable memory size for a given period
    
    Args:
        digitizer (object)
        period (float)
    Returns:
        memsize (int)
    """
    drate=digitizer.sample_rate()
    npoints = period    *drate
    e = int(np.ceil(np.log(npoints)/np.log(2)))
    #e +=1
    memsize = pow(2, e);
    digitizer.data_memory_size.set(2**e)
    if pre_trigger is None:
        spare=np.ceil((memsize-npoints)/16)*16
        pre_trigger=min(spare/2, 512)
        #pre_trigger=512
    digitizer.posttrigger_memory_size(memsize-pre_trigger)
    digitizer.pretrigger_memory_size(pre_trigger)
    if verbose:
        print('%s: sample rate %.3f Mhz, period %f [ms]'  % (digitizer.name, drate/1e6, period*1e3))
        print('%s: trace %d points, selected memsize %d'  % (digitizer.name, npoints, memsize))
        print('%s: pre and post trigger: %d %d'  % (digitizer.name,digitizer.pretrigger_memory_size(), digitizer.posttrigger_memory_size() ))
       

def measuresegment_m4i(digitizer,read_ch,  mV_range, period, Naverage=100, width=None, post_trigger=None, verbose=0):
    """ Measure block data with M4i
    
    Args:
        width (None or float): if a float, then process data
    Returns:
        data (numpy array)
    
    """
    if period is None:
        raise Exception('please set period for block measurements')
    select_digitizer_memsize(digitizer, period, verbose=verbose>=1)
    
    digitizer.initialize_channels(read_ch, mV_range=mV_range)
    dataraw = digitizer.blockavg_hardware_trigger_acquisition(mV_range=mV_range, nr_averages=Naverage, post_trigger=post_trigger)
    if isinstance(dataraw, tuple):
        dataraw=dataraw[0]
    data = np.transpose(np.reshape(dataraw,[-1,len(read_ch)]))
    # TO DO: Process data when several channels are used
    
    if verbose:
        print('measuresegment_m4i: processing data: width %s, data shape %s, memsize %s' % (width, data.shape, digitizer.data_memory_size() ) )
    if width is not None:
        samplerate=digitizer.sample_rate()
        pre_trigger=digitizer.pretrigger_memory_size()
        data, (r1, r2) = process_digitizer_trace(data.T, width, period, samplerate, padding=0,
              fig=300, pre_trigger=pre_trigger)
        if verbose:
            print('measuresegment_m4i: processing data: r1 %s, r2 %s' % (r1, r2) )
        data=data.T
    return data

def measuresegment(waveform, Naverage, station, minstrhandle, read_ch, mV_range=5000, period=None, sawtooth_width=None):
    try:
       isfpga = isinstance(minstrhandle, qtt.instrument_drivers.FPGA_ave.FPGA_ave)
    except:
       isfpga = False
    try:
       import qcodes.instrument_drivers.Spectrum.M4i
       ism4i = isinstance(minstrhandle, qcodes.instrument_drivers.Spectrum.M4i.M4i)                  
    except:
       ism4i = False
    if isfpga:
        ReadDevice = ['FPGA_ch%d' % c for c in read_ch]
        devicedata = minstrhandle.readFPGA(ReadDevice=ReadDevice, Naverage=Naverage)
        data_raw = [devicedata[ii] for ii in read_ch]
        data = np.vstack( [station.awg.sweep_process(d, waveform, Naverage) for d in data_raw])
    elif ism4i:
        post_trigger=minstrhandle.posttrigger_memory_size()
        data= measuresegment_m4i(minstrhandle, read_ch, mV_range, period, Naverage,
                                 width=sawtooth_width, post_trigger=post_trigger)
    else:
        raise Exception('Unrecognized fast readout instrument %s' % minstrhandle)
    return data

#%%
def scan2Dfast(station, scanjob, location=None, liveplotwindow=None, plotparam='measured', diff_dir=None, verbose=1):
    """Make a 2D scan and create qcodes dataset to store on disk.

    Args:
        station (object): contains all the instruments
        scanjob (scanjob_t): data for scan

    Returns:
        alldata (DataSet): contains the measurement data and metadata
    """
    gates = station.gates
    gatevals = gates.allvalues()
    
    if 'sd' in scanjob:
        warnings.warn('sd argument is not supported in scan2Dfast')

    if type(scanjob) is dict:
        warnings.warn('Use the scanjob_t class.', DeprecationWarning)
        scanjob = scanjob_t(scanjob)

    scanjob.parse_stepdata('stepdata')
    scanjob.parse_stepdata('sweepdata')

    minstrhandle = getattr(station, scanjob.get('minstrumenthandle', 'fpga'))

    read_ch = scanjob['minstrument']
    if isinstance(read_ch, int):
        read_ch = [read_ch]

    if isinstance(scanjob['stepdata']['param'], lin_comb_type) or isinstance(scanjob['sweepdata']['param'], lin_comb_type):
        scanjob['scantype'] = 'scan2Dfastvec'
        scanjob._start_end_to_range()
    else:
        scanjob['scantype'] = 'scan2Dfast'

    Naverage = scanjob.get('Naverage', 20)

    stepdata = scanjob['stepdata']
    sweepdata = scanjob['sweepdata']
    period = scanjob['sweepdata'].get('period', 1e-3)

    wait_time = stepdata.get('wait_time', 0)
    wait_time_startscan = scanjob.get('wait_time_startscan', 0)

    if scanjob['scantype'] == 'scan2Dfastvec':
        scanjob._parse_2Dvec()
        waveform, sweep_info = station.awg.sweep_gate_virt(sweepdata['param'], sweepdata['range'], period)
    else:
        scanjob._parse_2Dvec()
        sweeprange = (sweepdata['end'] - sweepdata['start'])    
        waveform, sweep_info = station.awg.sweep_gate(sweepdata['param'], sweeprange, period)
        sweepgate_value = (sweepdata['start'] + sweepdata['end']) / 2
        gates.set(sweepdata['param'], float(sweepgate_value))

    data = measuresegment(waveform, Naverage, station, minstrhandle, read_ch, period=period, sawtooth_width=waveform['width' ])
    if len(read_ch) == 1:
        measure_names = ['measured']
    else:
        measure_names = ['READOUT_ch%d' % c for c in read_ch]
        if plotparam == 'measured':
            plotparam = measure_names[0]

    scanvalues = scanjob._convert_scanjob_vec(station, data[0].shape[0])
    stepvalues = scanvalues[0]
    sweepvalues = scanvalues[1]

    logging.info('scan2D: %d %d' % (len(stepvalues), len(sweepvalues)))
    logging.info('scan2D: wait_time %f' % wait_time)

    t0 = qtt.time.time()

    alldata = makeDataSet2D(stepvalues, sweepvalues, measure_names=measure_names, location=location, loc_record={'label': scanjob['scantype']})

    # TODO: Allow liveplotting for multiple read-out channels
    if liveplotwindow is None:
        liveplotwindow = qtt.live.livePlot()
    if liveplotwindow is not None:
        liveplotwindow.clear()
        liveplotwindow.add(alldata.default_parameter_array(paramname=plotparam))

    tprev = time.time()

    for ix, x in enumerate(stepvalues):
        tprint('scan2Dfast: %d/%d: setting %s to %.3f' % (ix, len(stepvalues), stepvalues.name, x), dt=.5)
        if scanjob['scantype'] == 'scan2Dfastvec':
            for g in stepdata['param']:
                gates.set(g, (scanjob['phys_gates_vals'][g][ix, 0] + scanjob['phys_gates_vals'][g][ix, -1])/2)
        else:
            stepvalues.set(x)
        if ix == 0:
            qtt.time.sleep(wait_time_startscan)
        else:
            qtt.time.sleep(wait_time)
        data = measuresegment(waveform, Naverage, station, minstrhandle, read_ch, period=period, sawtooth_width=waveform['width' ])
        for idm, mname in enumerate(measure_names):
            alldata.arrays[mname].ndarray[ix] = data[idm]

        if liveplotwindow is not None:
            delta, tprev, update = delta_time(tprev, thr=2)
            if update:
                liveplotwindow.update_plot()
                pg.mkQApp().processEvents()
        if qtt.abort_measurements():
            print('  aborting measurement loop')
            break

    station.awg.stop()

    dt = qtt.time.time() - t0

    if diff_dir is not None:
        for mname in measure_names:
            alldata = diffDataset(alldata, diff_dir=diff_dir, fig=None, meas_arr_name=mname)

    if scanjob['scantype'] is 'scan2Dfastvec':
        for param in scanjob['phys_gates_vals']:
            parameter = gates.parameters[param]
            arr = DataArray(name=parameter.name, array_id=parameter.name, label=parameter.label, unit=parameter.unit, preset_data=scanjob['phys_gates_vals'][param], set_arrays=(alldata.arrays[stepvalues.parameter.name], alldata.arrays[sweepvalues.parameter.name]))
            alldata.add_array(arr)

    if not hasattr(alldata, 'metadata'):
        alldata.metadata = dict()

    update_dictionary(alldata.metadata, scanjob=scanjob, dt=dt, station=station.snapshot())
    update_dictionary(alldata.metadata, scantime=str(datetime.datetime.now()), allgatevalues=gatevals)

    alldata.write(write_metadata=True)

    return alldata


def plotData(alldata, diff_dir=None, fig=1):
    """ Plot a dataset and optionally differentiate """
    figure = plt.figure(fig)
    plt.clf()
    if diff_dir is not None:
        imx = qtt.diffImageSmooth(alldata.measured.ndarray, dy=diff_dir)
        name = 'diff_dir_%s' % diff_dir
        name = uniqueArrayName(alldata, name)
        data_arr = qcodes.DataArray(name=name, label=name, array_id=name, set_arrays=alldata.measured.set_arrays, preset_data=imx)
        alldata.add_array(data_arr)
        plot = MatPlot(interval=0, num=figure.number)
        plot.add(alldata.arrays[name])
        # plt.axis('image')
        plot.fig.axes[0].autoscale(tight=True)
        plot.fig.axes[1].autoscale(tight=True)
    else:
        plot = MatPlot(interval=0, num=figure.number)
        plot.add(alldata.default_parameter_array('measured'))
        # plt.axis('image')
        plot.fig.axes[0].autoscale(tight=True)
        try:
            # TODO: make this cleaner code
            plot.fig.axes[1].autoscale(tight=True)
        except:
            pass


#%%


def scan2Dturbo(station, scanjob, location=None, verbose=1):
    """Perform a very fast 2d scan by varying two physical gates with the AWG.

    The function assumes the station contains an FPGA with readFPGA function. 
        The number of the FPGA channel is supplied via the minstrument field in the scanjob.

    Args:
        station (object): contains all the instruments
        scanjob (scanjob_t): data for scan

    Returns:
        alldata (DataSet): contains the measurement data and metadata
    """
    gates = station.gates
    gatevals = gates.allvalues()
    
    if 'sd' in scanjob:
        warnings.warn('sd argument is not supported in scan2Dturbo')

    if type(scanjob) is dict:
        warnings.warn('Use the scanjob_t class.', DeprecationWarning)
        scanjob = scanjob_t(scanjob)

    scanjob.parse_stepdata('stepdata')
    scanjob.parse_stepdata('sweepdata')

    minstrhandle = getattr(station, scanjob.get('minstrumenthandle', 'fpga'))

    read_ch = scanjob['minstrument']
    if isinstance(read_ch, int):
        read_ch = [read_ch]

    if isinstance(scanjob['stepdata']['param'], lin_comb_type) or isinstance(scanjob['sweepdata']['param'], lin_comb_type):
        scanjob['scantype'] = 'scan2Dturbovec'
        scanjob._start_end_to_range()
    else:
        scanjob['scantype'] = 'scan2Dturbo'

    stepdata = scanjob['stepdata']
    sweepdata = scanjob['sweepdata']

    Naverage = scanjob.get('Naverage', 20)
    resolution = scanjob.get('resolution', [90, 90])

    t0 = qtt.time.time()

    wait_time_startscan = scanjob.get('wait_time_startscan', 0)

    if scanjob['scantype'] == 'scan2Dturbo':
        gates.set(stepdata['param'], (stepdata['end'] + stepdata['start']) / 2)
        gates.set(sweepdata['param'], (sweepdata['end'] + sweepdata['start']) / 2)
        sweepranges = [sweepdata['end'] - sweepdata['start'], stepdata['end'] - stepdata['start']]
    else:
        sweepranges = [sweepdata['range'], stepdata['range']]

    fpga_samp_freq = station.fpga.get_sampling_frequency()
    if scanjob['scantype'] == 'scan2Dturbo':
        sweepgates = [sweepdata['param'], stepdata['param']]
        waveform, sweep_info = station.awg.sweep_2D(fpga_samp_freq, sweepgates, sweepranges, resolution)
        if verbose:
            print('scan2Dturbo: sweepgates %s' % (str(sweepgates),))
    else:
        scanjob._parse_2Dvec()
        waveform, sweep_info = station.awg.sweep_2D_virt(fpga_samp_freq, sweepdata['param'], stepdata['param'], sweepranges, resolution)

    qtt.time.sleep(wait_time_startscan)

    waittime = resolution[0] * resolution[1] * Naverage / fpga_samp_freq
    ReadDevice = ['FPGA_ch%d' % c for c in read_ch]
    devicedata = station.fpga.readFPGA(Naverage=Naverage, ReadDevice=ReadDevice, waittime=waittime)
    station.awg.stop()
    data_raw = [devicedata[ii] for ii in read_ch]
    data = np.array([station.awg.sweep_2D_process(d, waveform) for d in data_raw])

    if len(read_ch) == 1:
        measure_names = ['measured']
    else:
        measure_names = ['FPGA_ch%d' % c for c in read_ch]
    
    if scanjob['scantype'] == 'scan2Dturbo':
        alldata, _ = makeDataset_sweep_2D(data, gates, sweepgates, sweepranges, measure_names=measure_names, location=location, loc_record={'label': scanjob['scantype']})
    else:
        scanvalues = scanjob._convert_scanjob_vec(station, data[0].shape[1], data[0].shape[0])
        stepvalues = scanvalues[0]
        sweepvalues = scanvalues[1]
        alldata = makeDataSet2D(stepvalues, sweepvalues, measure_names=measure_names, preset_data=data, location=location, loc_record={'label': scanjob['scantype']})

    dt = qtt.time.time() - t0

    if scanjob['scantype'] == 'scan2Dturbovec':
        for param in scanjob['phys_gates_vals']:
            parameter = gates.parameters[param]
            arr = DataArray(name=parameter.name, array_id=parameter.name, label=parameter.label, unit=parameter.unit, preset_data=scanjob['phys_gates_vals'][param], set_arrays=(alldata.arrays[stepvalues.parameter.name], alldata.arrays[sweepvalues.parameter.name]))
            alldata.add_array(arr)

    if not hasattr(alldata, 'metadata'):
        alldata.metadata = dict()

    update_dictionary(alldata.metadata, scanjob=scanjob, dt=dt, station=station.snapshot())
    update_dictionary(alldata.metadata, scantime=str(datetime.datetime.now()), allgatevalues=gatevals)

    alldata.write(write_metadata=True)

    return alldata

#%%


@deprecated
def scanLine(station, scangates, coords, sd, period=1e-3, Naverage=1000, verbose=1):
    ''' Do a scan (AWG sweep) over the line connecting two points.

    TODO: Add functionality for virtual gates, which should contain functionality to automatically determine
    whether to use the AWG or the IVVI's to scan. 

    Arguments:
        station (qcodes station): contains all of the instruments
        scangates (list of length k): the gates to scan
        coords (k x 2 array): coordinates of the points to scan between              
        sd (object): corresponds to the sensing dot used for read-out

    Returns:
        dataset (qcodes Dataset): measurement data and metadata
    '''
    # TODO: put a different parameter and values on the horizontal axis?
    # TODO: extend functionality to any number of gates (virtual gates?)
    # FIXME: single gate variation???
    x0 = [coords[0, 0], coords[0, 1]]  # first parameters
    x1 = [coords[1, 0], coords[1, 1]]
    sweeprange = np.sqrt((x1[1] - x1[0])**2 + (x0[1] - x0[0])**2)
    gate_comb = dict()

    # for g in scangates:
    #    gate_comb[g] = {scangates[1]: (x0[1] - x1[1]) / sweeprange, scangates[0]: (x0[0] - x1[0]) / sweeprange}
    gate_comb = {scangates[1]: (x1[1] - x1[0]) / sweeprange, scangates[0]: (x0[1] - x0[0]) / sweeprange}

    gate = scangates[0]  # see TODO: proper name

    waveform, sweep_info = station.awg.sweep_gate_virt(gate_comb, sweeprange, period)
    if verbose:
        print('scanLine: sweeprange %.1f ' % sweeprange)
        print(sweep_info)

    fpga_ch = sd.fpga_ch
    waittime = Naverage * period
    ReadDevice = ['FPGA_ch%d' % fpga_ch]
    _, DataRead_ch1, DataRead_ch2 = station.fpga.readFPGA(Naverage=Naverage, ReadDevice=ReadDevice, waittime=waittime)

    station.awg.stop()

    dataread = [DataRead_ch1, DataRead_ch2][fpga_ch - 1]
    data = station.awg.sweep_process(dataread, waveform, Naverage)
    dataset, _ = makeDataset_sweep(data, gate, sweeprange, gates=station.gates)  # see TODO

    dataset.write()

    return dataset

#%% Measurement tools


def waitTime(gate, station=None, gate_settle=None):
    """ Return settle times for gates on a station """
    if gate is None:
        return 0.001
    if gate_settle is not None:
        return gate_settle(gate)
    if station is not None:
        if hasattr(station, 'gate_settle'):
            return station.gate_settle(gate)
    return 0.001


def pinchoffFilename(g, od=None):
    ''' Return default filename of pinch-off scan '''
    if od is None:
        basename = 'pinchoff-sweep-1d-%s' % (g,)
    else:
        # old style filename
        basename = '%s-sweep-1d-%s' % (od['name'], g)
    return basename


def scanPinchValue(station, outputdir, gate, basevalues=None, minstrument=[1], stepdelay=None, cache=False, verbose=1, fig=10, full=0, background=False):
    basename = pinchoffFilename(gate, od=None)
    outputfile = os.path.join(outputdir, 'one_dot', basename + '.pickle')
    outputfile = os.path.join(outputdir, 'one_dot', basename)
    figfile = os.path.join(outputdir, 'one_dot', basename + '.png')

    if cache and os.path.exists(outputfile):
        if verbose:
            print('  cached data: skipping pinch-off scans for gate %s' % (gate))
            if verbose>=2:
                print(outputfile)
        alldata = qcodes.load_data(outputfile)
        return alldata

    if stepdelay is None:
        stepdelay = waitTime(gate, station=station)

    if basevalues is None:
        b = 0
    else:
        b = basevalues[gate]
    sweepdata = dict(
        {'param': gate, 'start': max(b, 0), 'end': -750, 'step': -2})
    if full == 0:
        sweepdata['step'] = -6

    scanjob = dict(
        {'sweepdata': sweepdata, 'minstrument': minstrument, 'wait_time': stepdelay})

    station.gates.set(gate, sweepdata['start'])  # set gate to starting value
    time.sleep(stepdelay)

    alldata = scan1D(station, scanjob=scanjob)

    station.gates.set(gate, basevalues[gate])  # reset gate to base value

    # show results
    if fig is not None:
        plot1D(alldata, fig=fig)

    adata = analyseGateSweep(alldata, fig=None, minthr=None, maxthr=None)
    alldata.metadata['adata'] = adata

    alldata = qtt.tools.stripDataset(alldata)
    writeDataset(outputfile, alldata)
    return alldata


#%%


def makeDataset_sweep(data, sweepgate, sweeprange, sweepgate_value=None,
                      ynames=None, gates=None, fig=None, location=None, loc_record=None):
    """Convert the data of a 1D sweep to a DataSet.

    Note: sweepvalues are only an approximation
    
     Args:
        data (1D array or kxN array)
        sweepgate (str)
        sweeprange (float)
        
    Returns:
        dataset

    """
    if sweepgate_value is None:
        if gates is not None:
            sweepgate_param = gates.getattr(sweepgate)
            sweepgate_value = sweepgate_param.get()
        else:
            raise Exception('No gates supplied')

    if isinstance(ynames, list):
        sweeplength = len(data[0])
    else:
        sweeplength = len(data)
    sweepvalues = np.linspace(sweepgate_value - sweeprange / 2, sweepgate_value + sweeprange / 2, sweeplength)

    if ynames is None:
        dataset = makeDataSet1Dplain(sweepgate, sweepvalues, yname='measured',
                                     y=data, location=location, loc_record=loc_record)
    else:
        dataset = makeDataSet1Dplain(sweepgate, sweepvalues, yname=ynames,
                                     y=data, location=location, loc_record=loc_record)

    if fig is None:
        return dataset, None
    else:
        plot = MatPlot(dataset.measured, interval=0, num=fig)
        return dataset, plot


def makeDataset_sweep_2D(data, gates, sweepgates, sweepranges, measure_names='measured', location=None, loc_record=None, fig=None):
    """Convert the data of a 2D sweep to a DataSet."""

    gate_horz = getattr(gates, sweepgates[0])
    gate_vert = getattr(gates, sweepgates[1])

    initval_horz = gate_horz.get()
    initval_vert = gate_vert.get()

    if type(measure_names) is list:
        data_measured = data[0]
    else:
        data_measured = data

    sweep_horz = gate_horz[initval_horz - sweepranges[0] /
                           2:sweepranges[0] / 2 + initval_horz:sweepranges[0] / len(data_measured[0])]
    sweep_vert = gate_vert[initval_vert - sweepranges[1] /
                           2:sweepranges[1] / 2 + initval_vert:sweepranges[1] / len(data_measured)]

    dataset = makeDataSet2D(sweep_vert, sweep_horz, measure_names=measure_names, location=location, loc_record=loc_record, preset_data=data)

    if fig is None:
        return dataset, None
    else:
        if fig is not None:
            plt.figure(fig).clear()
        plot = MatPlot(dataset.measured, interval=0, num=fig)
        return dataset, plot


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
    print('analyse data for 1-dot: %s' % od['name'])
    gg = od['gates']
    pv = np.zeros((3, 1))
    for ii, g in enumerate(gg):
        basename = pinchoffFilename(g, od=None)

        pfile = os.path.join(outputdir, 'one_dot', basename)
        alldata, mdata = loadDataset(pfile)
        if alldata is None:
            raise Exception('could not load file %s' % pfile)
        adata = analyseGateSweep(
            alldata, fig=None, minthr=None, maxthr=None, verbose=1)
        if verbose:
            print('loadOneDotPinchvalues: pinchvalue for gate %s: %.1f' %
                  (g, adata['pinchvalue']))
        pv[ii] = adata['pinchvalue']
    od['pinchvalues'] = pv
    return od


#%% Testing

from qcodes import ManualParameter
from qcodes.instrument_drivers.devices import VoltageDivider
from qtt.instrument_drivers.gates import virtual_IVVI
from qtt.instrument_drivers.virtual_instruments import VirtualIVVI

def test_scan2D(verbose=0):
    import qcodes
    import qtt.measurements.scans
    from qtt.measurements.scans import scanjob_t
    p = ManualParameter('p'); q = ManualParameter('q')
    R=VoltageDivider(p, 4)
    gates=VirtualIVVI(name=qtt.measurements.scans.instrumentName('gates'), model=None)
    station = qcodes.Station(gates)
    station.gates=gates

    if verbose:
        print('test_scan2D: running scan2D')
    scanjob = scanjob_t({'sweepdata': dict({'param':p, 'start': 0, 'end': 10, 'step': 2}), 'minstrument': [R], 'wait_time': 0.})
    scanjob['stepdata'] = dict({'param': q, 'start': 24, 'end': 30, 'step': 1.})
    data = scan2D(station, scanjob, liveplotwindow=False, verbose=0)

    scanjob = scanjob_t({'sweepdata': dict({'param': {'dac1': 1, 'dac2': .1}, 'start': 0, 'range': 10, 'step': 2}), 'minstrument': [R], 'wait_time': 0.})
    scanjob['stepdata'] = dict({'param': {'dac2': 1}, 'start': 24, 'range': 6, 'end': np.NaN, 'step': 1.})
    data = scan2D(station, scanjob, liveplotwindow=False, verbose=0)

    scanjob = scanjob_t({'sweepdata': dict({'param': p, 'start': 0, 'end': 10, 'step': 2}), 'minstrument': [R], 'wait_time': 0.})
    data = scan1D(station, scanjob, liveplotwindow=False, verbose=0)
 
    scanjob = scanjob_t({'sweepdata': dict({'param': 'dac1', 'start': 0, 'end': 10, 'step': 2}), 'minstrument': [R], 'wait_time': 0.})
    data = scan1D(station, scanjob, liveplotwindow=False, verbose=0)

    scanjob = scanjob_t({'sweepdata': dict({'param': {'dac1': 1}, 'start': 0, 'range': 10, 'step': 2}), 'minstrument': [R], 'wait_time': 0.})
    data = scan1D(station, scanjob, liveplotwindow=False, verbose=0)
    
    gates.close()



