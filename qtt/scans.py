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

import qtt.tools
from qtt.tools import tilefigs
from qtt.algorithms.gatesweep import analyseGateSweep
import qtt.algorithms.onedot  # import onedotGetBalanceFine
import qtt.live
from qtt.tools import deprecated

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


def createScanJob(g1, r1, g2=None, r2=None, step=-1, keithleyidx=[1]):
    """ Create a scan job

    Arguments
    ---------
    g1 : string
        Step gate
    r1 : array, list
        Range to step
    g2 : string, optional
        Sweep gate
    r2 : array, list
        Range to step
    step : integer, optional
        Step value

    """
    stepdata = dict(
        {'param': [g1], 'start': r1[0], 'end': r1[1], 'step': step})
    scanjob = dict({'stepdata': stepdata, 'minstrument': keithleyidx})
    if not g2 is None:
        sweepdata = dict(
            {'param': [g2], 'start': r2[0], 'end': r2[1], 'step': step})
        scanjob['sweepdata'] = sweepdata

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

    # TODO: implement range argument

    return stepdata


def get_param(gates, sweepgate):
    """ Get qcodes parameter from scanjob argument """
    if isinstance(sweepgate, str):
        return getattr(gates, sweepgate)
    else:
        # assume the argument already is a parameter
        return sweepgate


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

    wait_time = qtt.scans.waitTime(od['gates'][2], station=station)
    scanjobhi['sweepdata']['wait_time'] = wait_time
    scanjobhi['stepdata']['wait_time'] = qtt.scans.waitTime(None, station) + 3 * wait_time

    alldatahi = qtt.scans.scan2D(station, scanjobhi)
    extentscan, g0, g2, vstep, vsweep, arrayname = dataset2Dmetadata(
        alldatahi, verbose=0, arrayname=None)
    impixel, tr = dataset2image(alldatahi, mode='pixel')

    #_,_,_, im = get2Ddata(alldatahi)
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
    # saveExperimentData(outputdir, alldatahi, tag='one_dot', dstr='%s-sweep-2d-hires' % (od['name']))


if __name__ == '__main__':
    scandata, od = onedotHiresScan(station, od, dv=70, verbose=1)


#%%

from qcodes.plots.qcmatplotlib import MatPlot


def plot1D(data, fig=100, mstyle='-b'):
    """ Show result of a 1D gate scan """

    # kk=list(data.arrays.keys())

    val = data.default_parameter_name()

    if fig is not None:
        plt.figure(fig)
        plt.clf()
        MatPlot(getattr(data, val), interval=None, num=fig)
        # plt.show()


if __name__ == '__main__':
    plot1D(alldata, fig=100)

#%%


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
    scanjob['scantype'] = 'scan1D'

    gates = station.gates
    gatevals = gates.allvalues()
    sweepdata = parse_stepdata(scanjob['sweepdata'])
    gate = sweepdata.get('param', None)
    if gate is None:
        raise Exception('set param in scanjob')
    param = get_param(gates, gate)

    sweepvalues = param[sweepdata['start']:sweepdata['end']:sweepdata['step']]

    wait_time = sweepdata.get('wait_time', 0)
    wait_time_startscan = scanjob.get('wait_time_startscan', 0)
    t0 = time.time()

    # LEGACY
    instrument = scanjob.get('instrument', None)
    if instrument is not None:
        raise Exception('legacy argument instrument: use minstrument instead!')

    minstrument = scanjob.get('minstrument', None)
    mparams = get_measurement_params(station, minstrument)

    logging.debug('wait_time: %s' % str(wait_time))

    loop = qc.Loop(sweepvalues, delay=wait_time, progress_interval=1).each(*mparams)

    alldata = loop.get_data_set(location=location, loc_record={'label': 'scan1D'})

    if liveplotwindow is None:
        liveplotwindow = qtt.live.livePlot()
    if liveplotwindow is not None:
        liveplotwindow.clear()
        liveplotwindow.add(alldata.default_parameter_array(paramname=plotparam))

    if liveplotwindow is not None:
        def myupdate():
            t0 = time.time()
            liveplotwindow.update()
            # QtWidgets.QApplication.processEvents()
            if verbose >= 2:
                print('scan1D: myupdate: %.3f ' % (time.time() - t0))

        loop = loop.with_bg_task(myupdate, min_delay=1.8)

    param.set(sweepdata['start'])
    qtt.time.sleep(wait_time_startscan)
    alldata = loop.run()
    alldata.sync()
    dt = time.time() - t0

    if not hasattr(alldata, 'metadata'):
        alldata.metadata = dict()

    update_dictionary(alldata.metadata, scanjob=scanjob, dt=dt, station=station.snapshot())
    update_dictionary(alldata.metadata, scantime=str(datetime.datetime.now()), allgatevalues=gatevals)

    logging.info('scan1D: done %s' % (str(alldata.location),))

    alldata.write(write_metadata=True)

    return alldata


if __name__ == '__main__':

    loop3 = qc.Loop(gates.R[-15:15:1], 0.1).each(keithley1.amplitude)
    data = loop3.get_data_set(data_manager=False)

    reload(qtt.scans)
    scanjob = dict({'sweepdata': dict({'gate': 'R', 'start': -500, 'end': 1, 'step': .2}), 'instrument': [keithley3.amplitude], 'delay': .000})
    data1d = qtt.scans.scan1D(scanjob, station, location=None, background=None)

    data1d.sync()  # data.arrays


#%%
def scan1Dfast(station, scanjob, location=None, liveplotwindow=None, verbose=1):
    """Fast 1D scan. 

    Args:
        station (object): contains all data on the measurement station
        scanjob (scanjob_t): data for scan

    Returns:
        alldata (DataSet): contains the measurement data and metadata
    """
    scanjob['scantype'] = 'scan1Dfast'

    sweepdata = parse_stepdata(scanjob['sweepdata'])
    Naverage = scanjob.get('Naverage', 20)

    gates = station.gates
    gatevals = gates.allvalues()

    sweepgate = sweepdata['param']
    sweepparam = get_param(gates, sweepgate)

    if 'sd' in scanjob:
        warnings.warn('sd argument is not supported in scan1Dfast')
        
    minstrhandle = getattr(station, scanjob.get('minstrhandle', 'fpga'))
        
    read_ch = scanjob['minstrument']
    if isinstance(read_ch, int):
        read_ch = [read_ch]

    sweeprange = (sweepdata['end'] - sweepdata['start'])
    period = scanjob['sweepdata'].get('period', 1e-3)
    sweepgate_value = (sweepdata['start'] + sweepdata['end']) / 2

    t0 = qtt.time.time()

    waveform, sweep_info = station.awg.sweep_gate(sweepgate, sweeprange, period)

    sweepparam.set(float(sweepgate_value))
    wait_time_startscan = scanjob.get('wait_time_startscan', 0)
    qtt.time.sleep(wait_time_startscan)

    data = measuresegment(waveform, Naverage, station, minstrhandle, read_ch)
    alldata, _ = makeDataset_sweep(data, sweepgate, sweeprange, sweepgate_value=sweepgate_value,
                                   ynames=['measured%d' % i for i in read_ch],
                                   fig=None, location=location, loc_record={'label': 'scan1Dfast'})
    
    station.awg.stop()

    if liveplotwindow is None:
        liveplotwindow = qtt.live.livePlot()
    if liveplotwindow is not None:
        liveplotwindow.clear()
        liveplotwindow.add(alldata.default_parameter_array())

    dt = time.time() - t0

    if not hasattr(alldata, 'metadata'):
        alldata.metadata = dict()

    update_dictionary(alldata.metadata, scanjob=scanjob, dt=dt, station=station.snapshot())
    update_dictionary(alldata.metadata, scantime=str(datetime.datetime.now()), allgatevalues=gatevals)

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
    pass

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

def _convert_scanjob_vec(station, scanjob):
    """ Adjust the scanjob for vector scans. 
    
    Args:
        station (object): contains all the instruments
        scanjob (scanjob_t): data for scan
        
    Returns:
        scanjob (scanjob_t): updated data for scan
        scanvalues (array): contains the values for parameters to scan over
    """
    gates = station.gates

    stepdata = parse_stepdata(scanjob['stepdata'])
    sweepdata = parse_stepdata(scanjob['sweepdata'])

    if scanjob['scantype'] is 'scan2Dvec':
        params = set()
        vec_check = {'stepdata': isinstance(stepdata['param'], qtt.scans.lin_comb_type), 'sweepdata': isinstance(sweepdata['param'], qtt.scans.lin_comb_type)}
        for scaninfo in vec_check:
            if vec_check[scaninfo] is False:
                eval(scaninfo)['param'] = {eval(scaninfo)['param']: 1}
        params.update(list(stepdata['param'].keys()))
        params.update(list(sweepdata['param'].keys()))
        for param in params:
            if param not in stepdata['param']:
                stepdata['param'][param] = 0
            if param not in sweepdata['param']:
                sweepdata['param'][param] = 0
        stepname = 'stepparam'
        sweepname = 'sweepparam'
        if not (np.dot(list(stepdata['param'].values()), [sweepdata['param'][x] for x in stepdata['param'].keys()]) == 0):
            stepname = stepname + '*'
            sweepname= sweepname + '*'
        stepparam = VectorParameter(name=stepname, comb_map=[(gates.parameters[x], stepdata['param'][x]) for x in stepdata['param']])
        param = VectorParameter(name=sweepname, comb_map=[(gates.parameters[x], sweepdata['param'][x]) for x in sweepdata['param']])
    elif scanjob['scantype'] is 'scan2D':
        stepgate = scanjob['stepdata'].get('param', None)
        stepparam = get_param(gates, stepgate)
        sweepgate = scanjob['sweepdata'].get('param', None)
        param = get_param(gates, sweepgate)
    else:
        raise Exception('unknown scantype')
        
    sweepvalues = param[sweepdata['start']:sweepdata['end']:sweepdata['step']]
    stepvalues = stepparam[stepdata['start']:stepdata['end']:stepdata['step']]

    if scanjob['scantype'] is 'scan2Dvec':
        param_init = {param: gates.get(param) for param in params}
        scanjob['phys_gates_vals'] = {param: np.zeros((len(stepvalues), len(sweepvalues))) for param in params}
        step_array2d = np.tile(np.arange(0, stepdata['end']-stepdata['start'], stepdata['step'])[::-1].reshape((len(stepvalues)), 1), (1, len(sweepvalues)))
        sweep_array2d = np.tile(np.arange(0, sweepdata['end']-sweepdata['start'], sweepdata['step']), (len(stepvalues), 1))   
        for param in params:
            scanjob['phys_gates_vals'][param] = param_init[param] + step_array2d * stepdata['param'][param] + sweep_array2d * sweepdata['param'][param]

    scanvalues = [stepvalues, sweepvalues]

    scanjob['stepdata'] = stepdata
    scanjob['sweepdata'] = sweepdata

    return scanjob, scanvalues

lin_comb_type = dict 

def scan2D(station, scanjob, location=None, liveplotwindow=None, plotparam='measured', diff_dir=None, verbose=1):
    """Make a 2D scan and create dictionary to store on disk.

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

    if isinstance(scanjob['stepdata']['param'], qtt.scans.lin_comb_type) or isinstance(scanjob['sweepdata']['param'], qtt.scans.lin_comb_type):
        scanjob['scantype'] = 'scan2Dvec'
    else:
        scanjob['scantype'] = 'scan2D'

    scanjob, scanvalues = _convert_scanjob_vec(station, scanjob)
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

    # p.full_name
    if verbose >= 2:
        # print(alldata)
        print('scan2D: created dataset')
        print('  set_names: %s ' % (set_names,))
        print('  measure_names: %s ' % (measure_names,))

    t0 = qtt.time.time()

    if liveplotwindow is None:
        liveplotwindow = qtt.live.livePlot()
    if liveplotwindow is not None:
        liveplotwindow.clear()
        liveplotwindow.add(alldata.default_parameter_array(paramname=plotparam))

    tprev = time.time()
    for ix, x in enumerate(stepvalues):
        tprint('scan2D: %d/%d: time %.1f: setting %s to %.3f' % (ix, len(stepvalues), time.time() - t0, stepvalues.name, x), dt=1.5)
        if scanjob['scantype'] is 'scan2Dvec':
            pass
        else:
            stepvalues.set(x)
        for iy, y in enumerate(sweepvalues):
            if scanjob['scantype'] is 'scan2Dvec':
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
            if update and liveplotwindow is not None:
                liveplotwindow.update_plot()
                pg.mkQApp().processEvents()

        if qtt.abort_measurements():
            print('  aborting measurement loop')
            break
    dt = qtt.time.time() - t0

    if diff_dir is not None:
        alldata = diffDataset(alldata, diff_dir=diff_dir, fig=None)

    if not hasattr(alldata, 'metadata'):
        alldata.metadata = dict()

    if scanjob['scantype'] is 'scan2Dvec':
        for param in scanjob['phys_gates_vals']:
            parameter = gates.parameters[param]
            arr = DataArray(name=parameter.name, array_id=parameter.name, label=parameter.label, unit=parameter.unit, preset_data=scanjob['phys_gates_vals'][param], set_arrays=(alldata.stepparam, alldata.sweepparam))
            alldata.add_array(arr)

    update_dictionary(alldata.metadata, scanjob=scanjob, dt=dt, station=station.snapshot())
    update_dictionary(alldata.metadata, scantime=str(datetime.datetime.now()), allgatevalues=gatevals)

    alldata.write(write_metadata=True)

    return alldata


if __name__ == '__main__':
    import logging
    import datetime
    scanjob['stepdata'] = dict({'gate': 'L', 'start': -340, 'end': 250, 'step': 3.})
    data = scan2D(station, scanjob, background=True, verbose=2, liveplotwindow=plotQ)

#%%

def process_digitizer_trace(data, width, period, samplerate, padding=0):
    """ Process data from the M4i and a sawtooth trace 
    
    This is done to remove the extra padded data of the digitized and to 
    extract the forward trace of the sawtooth.
    
    Args:
        data (Nxk array)
        width (float): with of the sawtooth
        period (float)
        samplerate (float)
    Returns
        processed_data (Nxk array): processed data
        rr (tuple)
    """
    npoints = period    *samplerate # expected number of points

    npoints2=width*npoints
    npoints2=npoints2-(npoints2%2)
    r1=int(data.shape[0]/2-npoints2/2)-padding
    r2=int(data.shape[0]/2+npoints2/2)+padding
    processed_data=data[ r1:r2,:]
    return processed_data, (r1, r2)

def select_digitizer_memsize(digitizer, period, verbose=1):
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
    memsize = pow(2, np.ceil(np.log(npoints)/np.log(2)));
    if verbose:
        print('%s: sample rate %.3f Mhz, period %f [ms]'  % (digitizer.name, drate/1e6, period/1e3))
        print('%s: trace %d points, selected memsize %d'  % (digitizer.name, npoints, memsize))
    digitizer.data_memory_size.set(2**e)


def measuresegment(waveform, Naverage, station, minstrhandle, read_ch, mV_range=1000):
#    if isinstance(minstrhandle, qtt.instrument_drivers.FPGA_ave):
    if minstrhandle.name == 'fpga':
        ReadDevice = ['FPGA_ch%d' % c for c in read_ch]
        devicedata = minstrhandle.readFPGA(ReadDevice=ReadDevice, Naverage=Naverage)
        data_raw = [devicedata[ii] for ii in read_ch]
        data = np.vstack( [station.awg.sweep_process(d, waveform, Naverage) for d in data_raw])
#    elif isinstance(minstrhandle, qcodes.instrument_drivers.Spectrum.M4i):
    elif minstrhandle.name == 'digitizer':
        minstrhandle.initialize_channels(read_ch, mV_range=mV_range)
        dataraw = minstrhandle.blockavg_hardware_trigger_acquisition(mV_range=mV_range, nr_averages=Naverage)
        if isinstance(dataraw, tuple):
            dataraw=dataraw[0]
        data = np.transpose(np.reshape(dataraw,[-1,len(read_ch)]))
#        data = np.vstack([datatemp])
        # TO DO: Process data when several channels are used
    else:
        raise Exception('Unrecognized fast readout instrument')
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
    scanjob['scantype'] = 'scan2Dfast'

    stepdata = scanjob['stepdata']
    sweepdata = scanjob['sweepdata']
    Naverage = scanjob.get('Naverage', 20)

    wait_time = stepdata.get('wait_time', 0)
    wait_time_startscan = scanjob.get('wait_time_startscan', 0)

    gates = station.gates
    gatevals = gates.allvalues()

    sweepgate = sweepdata.get('param', None)
    sweepparam = get_param(gates, sweepgate)

    stepgate = stepdata.get('param', None)
    stepparam = get_param(gates, stepgate)

    if 'sd' in scanjob:
        warnings.warn('sd argument is not supported in scan2Dfast')
    
    minstrhandle = getattr(station, scanjob.get('minstrhandle', 'fpga'))
    
    read_ch = scanjob['minstrument']
    if isinstance(read_ch, int):
        read_ch = [read_ch]

    sweeprange = (sweepdata['end'] - sweepdata['start'])
    period = scanjob['sweepdata'].get('period', 1e-3)
    sweepgate_value = (sweepdata['start'] + sweepdata['end']) / 2

    if 'gates_horz' in scanjob:
        waveform, sweep_info = station.awg.sweep_gate_virt(scanjob['gates_horz'], sweeprange, period)
    else:
        waveform, sweep_info = station.awg.sweep_gate(sweepgate, sweeprange, period)

    if 'gates_vert' in scanjob:
        scanjob['gates_vert_init'] = {}
        for g in scanjob['gates_vert']:
            gates.set(g, gates.get(g) + (stepdata['start'] - stepdata['end']) * scanjob['gates_vert'][g] / 2)
            scanjob['gates_vert_init'][g] = gates.get(g)
    else:
        sweepparam.set(float(sweepgate_value))

    data = measuresegment(waveform, Naverage, station, minstrhandle, read_ch)
    if len(read_ch) == 1:
        measure_names = ['measured']
    else:
        measure_names = ['READOUT_ch%d' % c for c in read_ch]
        if plotparam == 'measured':
            plotparam = measure_names[0]

    ds0, _ = makeDataset_sweep(data, sweepgate, sweeprange, sweepgate_value=sweepgate_value, ynames=measure_names, fig=None)

    sweepvalues = sweepparam[list(ds0.arrays[sweepgate])]
    stepvalues = stepparam[stepdata['start']:stepdata['end']:stepdata['step']]

    logging.info('scan2D: %d %d' % (len(stepvalues), len(sweepvalues)))
    logging.info('scan2D: wait_time %f' % wait_time)

    t0 = qtt.time.time()

    alldata = makeDataSet2D(stepvalues, sweepvalues, measure_names=measure_names, location=location, loc_record={'label': 'scan2Dfast'})

    # TODO: Allow liveplotting for multiple read-out channels
    if liveplotwindow is None:
        liveplotwindow = qtt.live.livePlot()
    if liveplotwindow is not None:
        liveplotwindow.clear()
        liveplotwindow.add(alldata.default_parameter_array(paramname=plotparam))

    tprev = time.time()

    for ix, x in enumerate(stepvalues):
        tprint('scan2Dfast: %d/%d: setting %s to %.3f' % (ix, len(stepvalues), stepvalues.name, x), dt=.5)
        if 'gates_vert' in scanjob:
            for g in scanjob['gates_vert']:
                gates.set(g, scanjob['gates_vert_init'][g] + ix * stepdata['step'] * scanjob['gates_vert'][g])
        else:
            stepvalues.set(x)
        if ix == 0:
            qtt.time.sleep(wait_time_startscan)
        else:
            qtt.time.sleep(wait_time)
        data = measuresegment(waveform, Naverage, station, minstrhandle, read_ch)
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
    scanjob['scantype'] = 'scan2Dturbo'
    gatevals = station.gates.allvalues()
    stepdata = parse_stepdata(scanjob['stepdata'])
    sweepdata = parse_stepdata(scanjob['sweepdata'])

    sweepgates = [sweepdata['param'], stepdata['param']]
    sweepranges = [sweepdata['end'] - sweepdata['start'], stepdata['end'] - stepdata['start']]

    if 'sd' in scanjob:
        warnings.warn('sd argument is not supported in scan2Dturbo')

    Naverage = scanjob.get('Naverage', 20)
    resolution = scanjob.get('resolution', [90, 90])

    if verbose:
        print('scan2Dturbo: sweepgates %s' % (str(sweepgates),))

    fpga_ch = scanjob['minstrument']
    if isinstance(fpga_ch, int):
        fpga_ch = [fpga_ch]

    fpga_samp_freq = station.fpga.get_sampling_frequency()

    t0 = qtt.time.time()

    station.gates.set(stepdata['param'], (stepdata['end'] + stepdata['start']) / 2)
    station.gates.set(sweepdata['param'], (sweepdata['end'] + sweepdata['start']) / 2)

    wait_time_startscan = scanjob.get('wait_time_startscan', 0)

    waveform, sweep_info = station.awg.sweep_2D(fpga_samp_freq, sweepgates, sweepranges, resolution)
    qtt.time.sleep(wait_time_startscan)
    waittime = resolution[0] * resolution[1] * Naverage / fpga_samp_freq

    ReadDevice = ['FPGA_ch%d' % c for c in fpga_ch]
    devicedata = station.fpga.readFPGA(Naverage=Naverage, ReadDevice=ReadDevice, waittime=waittime)
    station.awg.stop()
    data_raw = [devicedata[ii] for ii in fpga_ch]
    data = np.array([station.awg.sweep_2D_process(d, waveform) for d in data_raw])

    if len(fpga_ch) == 1:
        measure_names = ['measured']
    else:
        measure_names = ['FPGA_ch%d' % c for c in fpga_ch]
    alldata, _ = makeDataset_sweep_2D(data, station.gates, sweepgates, sweepranges, measure_names=measure_names, location=location, loc_record={'label': 'scan2Dturbo'})

    dt = qtt.time.time() - t0

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
        print('  skipping pinch-off scans for gate %s' % (gate))
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
        # plt.savefig(figfile)

    adata = analyseGateSweep(alldata, fig=None, minthr=None, maxthr=None)
    alldata.metadata['adata'] = adata
    #  alldata['adata'] = adata

    alldata = qtt.tools.stripDataset(alldata)
    writeDataset(outputfile, alldata)
    # alldata.write_to_disk(outputfile)
 #   pmatlab.save(outputfile, alldata)
    return alldata


if __name__ == '__main__':
    gate = 'L'
    alldataX = qtt.scans.scanPinchValue(
        station, outputdir, gate, basevalues=basevalues, keithleyidx=[3], cache=cache, full=full)
    adata = analyseGateSweep(alldataX, fig=10, minthr=None, maxthr=None)

#%%
from qtt.data import makeDataSet1D, makeDataSet2D, makeDataSet1Dplain

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
        # alldata,=pmatlab.load(pfile);  # alldata=alldata[0]
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

#%%


#%% Testing

from qcodes import ManualParameter
from qcodes.instrument_drivers.devices import VoltageDivider
from qtt.instrument_drivers.gates import virtual_IVVI

def test_scan2D():
    p = ManualParameter('p'); q = ManualParameter('q')
    R=VoltageDivider(p, 4)
    gates=virtual_IVVI(name='gates', instruments=[], gate_map={})
    station = qcodes.Station(gates)

    scanjob = scanjob_t({'sweepdata': dict({'param':p, 'start': 0, 'end': 10, 'step': 2}), 'minstrument': [R], 'wait_time': 0.})
    scanjob['stepdata'] = dict({'param': q, 'start': 24, 'end': 30, 'step': 1.})
    data = scan2D(station, scanjob, liveplotwindow=None, verbose=0)
    
if __name__ == '__main__':
    import qtt.scans
    reload(qtt.scans)
    od = qtt.scans.loadOneDotPinchvalues(od, outputdir, verbose=1)


#%%


if __name__ == '__main__':
    # ,'SD1a', 'SD1b', ''SD2a','SD]:
    for gate in ['L', 'D1', 'D2', 'D3', 'R'] + ['P1', 'P2', 'P3', 'P4']:
        alldata = scanPinchValue(station, outputdir, gate, basevalues=basevalues, keithleyidx=[
                                 3], cache=cache, full=full)
