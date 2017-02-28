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
import warnings
import pyqtgraph as pg

import matplotlib.pyplot as plt

from qtt.tools import tilefigs
import qtt.tools
from qtt.algorithms.gatesweep import analyseGateSweep
import qtt.algorithms.onedot  # import onedotGetBalanceFine
import qtt.live

#from qtt.data import *
from qtt.data import diffDataset, experimentFile, loadDataset, writeDataset
from qtt.data import uniqueArrayName
from qcodes.utils.helpers import tprint

from qtt.tools import update_dictionary

#%%

import skimage
import skimage.filters


def checkReversal(im0, verbose=0):
    """ Check sign of a current scan

    We assume that the current is either zero or positive 
    Needed when the keithley (or some other measurement device) has been reversed
    """
    thr = skimage.filters.threshold_otsu(im0)
    mval = np.mean(im0)

    # meanopen = np.mean(im0[:,:])
    fr=thr<mval
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
        {'gates': [g1], 'start': r1[0], 'end': r1[1], 'step': step})
    scanjob = dict({'stepdata': stepdata, 'keithleyidx': keithleyidx})
    if not g2 is None:
        sweepdata = dict(
            {'gates': [g2], 'start': r2[0], 'end': r2[1], 'step': step})
        scanjob['sweepdata'] = sweepdata

    return scanjob

#%%

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
    scanjobhi['keithleyidx'] = keithleyidx
    scanjobhi['stepdata']['end'] = max(scanjobhi['stepdata']['end'], -780)
    scanjobhi['sweepdata']['end'] = max(scanjobhi['sweepdata']['end'], -780)

    wait_time = qtt.scans.waitTime(od['gates'][2], station=station)
    scanjobhi['wait_time_step'] = qtt.scans.waitTime(None, station) + 3 * wait_time

    alldatahi = qtt.scans.scan2D(station, scanjobhi, wait_time=wait_time, background=False)
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

import time


def getParams(station, keithleyidx):
    """ Get qcodes parameters from an index or string """
    params = []
    for x in keithleyidx:
        if isinstance(x, int):
            params += [getattr(station, 'keithley%d' % x).amplitude]
        else:
            if isinstance(x, str):
                params += [getattr(station, x).amplitude]
            else:
                params += [x]
    return params


def getDefaultParameter(data):
    return data.default_parameter_name()

#%%


def scan1D(station, scanjob, location=None, liveplotwindow=None, background=False, title_comment=None, wait_time=None, verbose=1):
    """ Simple 1D scan. 

    Args:
        station (object): contains all data on the measurement station
        scanjob (dict): data for scan range
    """
    gates = station.gates
    sweepdata = scanjob['sweepdata']
    gate = sweepdata.get('gate', None)
    if gate is None:
        gate = sweepdata.get('gates')[0]
    param = getattr(gates, gate)
    sweepvalues = param[sweepdata['start']:sweepdata['end']:sweepdata['step']]

    scanjob['scantype'] = 'scan1D'

    if wait_time is None:
        wait_time = scanjob.get('wait_time', 0)
    t0 = time.time()

    # legacy code...
    minstrument = scanjob.get('instrument', None)
    if minstrument is None:
        minstrument = scanjob.get('keithleyidx', None)
    params = getParams(station, minstrument)

    # station.set_measurement(*params)

    if background:
        data_manager = True
    else:
        data_manager = False
        background = False

    if liveplotwindow is None:
        liveplotwindow = qtt.live.livePlot()

    logging.debug('wait_time: %s' % str(wait_time))
    if verbose:
        print('scan1D: starting Loop (background %s)' % background)

    if background:
        if verbose >= 2:
            print('scan1D: background %s, data_manager %s' % (background, data_manager))
        loop = qc.Loop(sweepvalues, delay=wait_time, progress_interval=1).each(*params)
        loop.background_functions = dict({'qt': pg.mkQApp().processEvents})
        alldata = loop.run(location=location, data_manager=data_manager, background=background)
        if liveplotwindow is not None:
            time.sleep(.1)
            alldata.sync()  # wait for at least 1 data point
            liveplotwindow.clear()
            liveplotwindow.add(getDefaultParameter(data))
        alldata.complete(delay=.25)
        alldata.sync()
        dt = -1
        if qcodes.get_bg() is not None:
            logging.info('background measurement not completed')
            time.sleep(.1)

        logging.info('scan1D: running %s' % (str(alldata.location),))
    else:
        # run with live plotting loop
        loop = qc.Loop(sweepvalues, delay=wait_time, progress_interval=1).each(*params)
        print('loop.data_set: %s' % loop.data_set)
        print('background: %s, data_managr %s' % (background, data_manager))
        alldata = loop.get_data_set(data_manager=data_manager, location=location)

        if liveplotwindow is not None:
            liveplotwindow.clear()
            liveplotwindow.add(alldata.default_parameter_array())

        dt = time.time() - t0
        if liveplotwindow is not None:
            def myupdate():
                t0 = time.time()
                liveplotwindow.update()
                # QtWidgets.QApplication.processEvents()
                if verbose >= 2:
                    print('scan1D: myupdate: %.3f ' % (time.time() - t0))

            alldata = loop.with_bg_task(myupdate, min_delay=1.8).run(background=background)
        else:
            alldata = loop.run(background=background)
        alldata.sync()

    if not hasattr(alldata, 'metadata'):
        alldata.metadata = dict()

    # add metadata
    metadata = alldata.metadata
    metadata['allgatevalues'] = gates.allvalues()
    metadata['scantime'] = str(datetime.datetime.now())
    metadata['dt'] = dt
    metadata['scanparams'] = {'wait_time': wait_time}
    metadata['scanjob'] = scanjob
    metadata['scanjob']['instrument'] = 'dummy'  # FIXME

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
def scan1Dfast(station, scanjob, wait_time=None, verbose=1):
    """ 1D scan with AWG. 

    Args:
        station (object): contains all data on the measurement station
        scanjob (dict): data for scan range

    Returns:
        alldata (DataSet)
    """
    scanjob['scantype'] = 'scan1Dfast'

    sweepdata = scanjob['sweepdata']
    Naverage = scanjob.get('Naverage', 20)

    if wait_time is None:
        wait_time = scanjob.get('wait_time', 0.5)

    gates = station.gates
    gvs = gates.allvalues()

    sweepgate = sweepdata.get('gate', None)
    if sweepgate is None:
        sweepgate = sweepdata.get('gates')[0]
    sweepparam = getattr(gates, sweepgate)

    def readfunc(waveform, Naverage):
        fpga_ch = scanjob['sd'].fpga_ch
        ReadDevice = ['FPGA_ch%d' % fpga_ch]
        data_raw = np.array(station.fpga.readFPGA(ReadDevice=ReadDevice, Naverage=Naverage)[fpga_ch])
        data = station.awg.sweep_process(data_raw, waveform, Naverage)
        return data

    sweeprange = (sweepdata['end'] - sweepdata['start'])
    period = scanjob['sweepdata'].get('period', 1e-3)
    sweepgate_value = (sweepdata['start'] + sweepdata['end']) / 2

    waveform, sweep_info = station.awg.sweep_gate(sweepgate, sweeprange, period)

    sweepparam.set(float(sweepgate_value))

    data = readfunc(waveform, Naverage)
    alldata, _ = makeDataset_sweep(data, sweepgate, sweeprange, sweepgate_value=sweepgate_value, fig=None)

    station.awg.stop()

    if not hasattr(alldata, 'metadata'):
        alldata.metadata = dict()

    # add the station metadata
    alldata.add_metadata({'station': station.snapshot()})

    alldata.metadata['scanjob'] = scanjob
    alldata.metadata['allgatevalues'] = gvs
    alldata.metadata['scantime'] = str(datetime.datetime.now())
    alldata.metadata['wait_time'] = wait_time

    alldata.write(write_metdata=True)

    return alldata

#%%


def wait_bg_finish(verbose=0):
    """ Wait for background job to finish """
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


def scan2Dold(station, scanjob, title_comment='', liveplotwindow=None, wait_time=None, background=False, verbose=1):
    """ Make a 2D scan and create dictionary to store on disk.

    Args:
        station (object): contains all data on the measurement station
        scanjob (dict): data for scan range
    """

    stepdata = scanjob['stepdata']
    sweepdata = scanjob['sweepdata']
    minstrument = scanjob.get('instrument', None)
    if minstrument is None:
        minstrument = scanjob.get('keithleyidx', None)

    if verbose:
        logging.info('scan2D: todo: implement compensategates')
        logging.info('scan2D: todo: implement wait_time')

    delay = scanjob.get('delay', 0.0)
    if wait_time is not None:
        delay = wait_time

    gates = station.gates

    sweepgate = sweepdata.get('gate', None)
    if sweepgate is None:
        sweepgate = sweepdata.get('gates')[0]

    stepgate = stepdata.get('gate', None)
    if stepgate is None:
        stepgate = stepdata.get('gates')[0]
    param = getattr(gates, sweepgate)
    stepparam = getattr(gates, stepgate)

    sweepvalues = param[sweepdata['start']:sweepdata['end']:sweepdata['step']]
    stepvalues = stepparam[stepdata['start']:stepdata['end']:stepdata['step']]

    logging.info('scan2D: %d %d' % (len(stepvalues), len(sweepvalues)))
    logging.info('scan2D: delay %f' % delay)
    steploop = qc.Loop(stepvalues, delay=delay, progress_interval=2)

    t0 = time.time()
    fullloop = steploop.loop(sweepvalues, delay=delay)

    params = getParams(station, minstrument)
    loop = fullloop.each(*params)

    if background is None:
        try:
            gates._server_name
            background = True
        except:
            background = False

    if liveplotwindow is None:
        liveplotwindow = qtt.live.livePlot()

    if background:
        if verbose >= 2:
            print('background %s, data_manager %s' % (background, '[default]'))
        # alldata=loop.get_data_set() # use default data_manager
        alldata = loop.run(background=background, data_manager=True)

        if liveplotwindow is not None:
            liveplotwindow.clear()
            liveplotwindow.add(getDefaultParameter(alldata))
        alldata.background_functions = dict({'qt': pg.mkQApp().processEvents})
        alldata.complete(delay=.5)
        #print('complete: %.3f' % alldata.fraction_complete() )
        wait_bg_finish(verbose=verbose >= 2)
    else:
        data_manager = False
        alldata = loop.get_data_set(data_manager=data_manager)

        if liveplotwindow is not None:
            liveplotwindow.clear()
            liveplotwindow.add(getDefaultParameter(alldata))

        def myupdate():
            t0 = time.time()
            liveplotwindow.update()
            # plt.pause(1e-5)

            # QtWidgets.QApplication.processEvents()
            if verbose >= 2:
                print('myupdate: %.3f ' % (time.time() - t0))

        alldata = loop.with_bg_task(myupdate, min_delay=.4).run(background=background)

    dt = time.time() - t0

    if not hasattr(alldata, 'metadata'):
        alldata.metadata = dict()

    alldata.add_metadata({'station': station.snapshot()})

    alldata.metadata['scanjob'] = scanjob
    alldata.metadata['allgatevalues'] = gates.allvalues()
    alldata.metadata['scantime'] = str(datetime.datetime.now())
    alldata.metadata['dt'] = dt
    alldata.metadata['wait_time'] = wait_time

    alldata.write()

    return alldata


def delta_time(tprev, thr=2):
    """ Helper function to prevent too many updates """
    t = time.time()
    delta = t - tprev
    if delta > thr:
        tprev = t
    return delta, tprev


def scan2D(station, scanjob, liveplotwindow=None, wait_time=None, background=False, diff_dir=None, verbose=1):
    """ Make a 2D scan and create dictionary to store on disk

    Args:
        station (object): contains all data on the measurement station
        scanjob (dict): data for scan range

    Returns:
        alldata (DataSet)
    """

    scanjob['scantype'] = 'scan2D'

    stepdata = scanjob['stepdata']
    sweepdata = scanjob['sweepdata']
    minstrument = scanjob.get('instrument', None)
    if minstrument is None:
        minstrument = scanjob.get('keithleyidx', None)

    gates = station.gates
    # get the gates to step and sweep
    sweepgate = sweepdata.get('gate', None)
    if sweepgate is None:
        sweepgate = sweepdata.get('gates')[0]

    stepgate = stepdata.get('gate', None)
    if stepgate is None:
        stepgate = stepdata.get('gates')[0]
    param = getattr(gates, sweepgate)
    stepparam = getattr(gates, stepgate)

    sweepvalues = param[sweepdata['start']:sweepdata['end']:sweepdata['step']]
    stepvalues = stepparam[stepdata['start']:stepdata['end']:stepdata['step']]

    # get the delays
    if wait_time is None:
        wait_time = scanjob.get('wait_time', None)
        if wait_time is None:
            wait_time = waitTime(sweepgate) / 2.
            wait_time = waitTime(sweepgate, station=station) / 8.

    wait_time_step = scanjob.get('wait_time_step', 4 * wait_time)
    logging.info('scan2D: %d %d' % (len(stepvalues), len(sweepvalues)))
    logging.info('scan2D: wait_time %f' % wait_time)

    params = getParams(station, minstrument)

    alldata = makeDataSet2D(stepvalues, sweepvalues)

    t0 = qtt.time.time()

    if background:
        warnings.warn('scan2D: background running not implemented, running in foreground')

    if liveplotwindow is None:
        liveplotwindow = qtt.live.livePlot()
    if liveplotwindow is not None:
        liveplotwindow.clear()
        liveplotwindow.add(alldata.default_parameter_array(paramname='measured'))

    tprev = time.time()
    for ix, x in enumerate(stepvalues):
        tprint('scan2D: %d/%d: time %.1f: setting %s to %.3f' % (ix, len(stepvalues), time.time() - t0, stepvalues.name, x), dt=.5)
        if 'gates_vert' in scanjob:
            for g in scanjob['gates_vert']:
                gates.set(g, scanjob['gates_vert_init'][g] + ix * stepdata['step'] * scanjob['gates_vert'][g])
        else:
            stepvalues.set(x)

        for iy, y in enumerate(sweepvalues):
            sweepvalues.set(y)
            if iy == 0:
                qtt.time.sleep(wait_time_step)
            time.sleep(wait_time)

            value = params[0].get()
            alldata.measured.ndarray[ix, iy] = value

        if ix == len(stepvalues) - 1 or ix % 5 == 0:
            delta, tprev = delta_time(tprev, thr=2)
            if delta > 2 and liveplotwindow is not None:
                liveplotwindow.update_plot()
                pg.mkQApp().processEvents()

    dt = qtt.time.time() - t0

    if not hasattr(alldata, 'metadata'):
        alldata.metadata = dict()

    if diff_dir is not None:
        alldata = diffDataset(alldata, diff_dir=diff_dir, fig=None)

    alldata.add_metadata({'station': station.snapshot()})

    update_dictionary(alldata.metadata, scanjob=scanjob, dt=dt,
                      scantime=str(datetime.datetime.now()), wait_time=wait_time)
    update_dictionary(alldata.metadata, allgatevalues = gates.allvalues(), scanparams={'wait_time': wait_time} )
    

    alldata.write(write_metadata=True)
    # print(type(scanjob['stepdata']['start']))
    # print(type(alldata.metadata['scanjob']['stepdata']['start']))
    # return alldata, alldata.metadata['scanjob']['stepdata']['start']

    return alldata

if __name__ == '__main__':
    import logging
    import datetime
    scanjob['stepdata'] = dict({'gate': 'L', 'start': -340, 'end': 250, 'step': 3.})
    data = scan2D(station, scanjob, background=True, verbose=2, liveplotwindow=plotQ)


#%%
def scan2Dfast(station, scanjob, liveplotwindow=None, wait_time=None, background=None, diff_dir=None, verbose=1):
    """ Make a 2D scan and create qcodes dataset to store on disk.

    Args:
        station (qcodes.station.Station): contains all data on the measurement station
        scanjob (dict): data for scan range

    Returns:
        alldata (qcodes.data.data_set.DataSet): measurement data and metadata
    """
    scanjob['scantype'] = 'scan2Dfast'

    stepdata = scanjob['stepdata']
    sweepdata = scanjob['sweepdata']
    Naverage = scanjob.get('Naverage', 20)

    if wait_time is None:
        wait_time = scanjob.get('wait_time', 0.5)

    wait_time_startloop = scanjob.get('wait_time_startloop', 2.0 + 4 * wait_time)

    gates = station.gates
    gvs = gates.allvalues()

    sweepgate = sweepdata.get('gate', None)
    if sweepgate is None:
        sweepgate = sweepdata.get('gates')[0]
    sweepparam = getattr(gates, sweepgate)

    stepgate = stepdata.get('gate', None)
    if stepgate is None:
        stepgate = stepdata.get('gates')[0]

    stepparam = getattr(gates, stepgate)

    def readfunc(waveform, Naverage):
        fpga_ch = scanjob['sd'].fpga_ch
        ReadDevice = ['FPGA_ch%d' % fpga_ch]
        data_raw = np.array(station.fpga.readFPGA(ReadDevice=ReadDevice, Naverage=Naverage)[fpga_ch])
        data = station.awg.sweep_process(data_raw, waveform, Naverage)
        return data

    sweeprange = (sweepdata['end'] - sweepdata['start'])
    # sweeprange = qtt.algorithms.generic.signedmin(sweeprange, 60)  # FIXME
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
        stepparam.set(stepdata['start'])
        sweepparam.set(float(sweepgate_value))

    qtt.time.sleep(wait_time_startloop)

    data = readfunc(waveform, Naverage)
    ds0, _ = makeDataset_sweep(data, sweepgate, sweeprange, sweepgate_value=sweepgate_value, fig=None)

    sweepvalues = sweepparam[list(ds0.arrays[sweepgate])]
    stepvalues = stepparam[stepdata['start']:stepdata['end']:stepdata['step']]

    logging.info('scan2D: %d %d' % (len(stepvalues), len(sweepvalues)))
    logging.info('scan2D: wait_time %f' % wait_time)

    t0 = qtt.time.time()

    if background:
        warnings.warn('scan2D: background running not implemented, running in foreground')

    alldata = makeDataSet2D(stepvalues, sweepvalues)

    if liveplotwindow is None:
        liveplotwindow = qtt.live.livePlot()
    if liveplotwindow is not None:
        liveplotwindow.clear()
        liveplotwindow.add(alldata.default_parameter_array(paramname='measured'))

    tprev = time.time()

    for ix, x in enumerate(stepvalues):
        tprint('scan2Dfast: %d/%d: setting %s to %.3f' % (ix, len(stepvalues), stepvalues.name, x), dt=.5)
        if 'gates_vert' in scanjob:
            for g in scanjob['gates_vert']:
                gates.set(g, scanjob['gates_vert_init'][g] + ix * stepdata['step'] * scanjob['gates_vert'][g])
        else:
            stepvalues.set(x)
        qtt.time.sleep(wait_time)
        alldata.measured.ndarray[ix] = readfunc(waveform, Naverage)
        if liveplotwindow is not None:
            delta, tprev = delta_time(tprev, thr=2)
            if delta > 2:
                liveplotwindow.update_plot()
                pg.mkQApp().processEvents()

    station.awg.stop()

    dt = qtt.time.time() - t0

    if not hasattr(alldata, 'metadata'):
        alldata.metadata = dict()

    if diff_dir is not None:
        alldata = diffDataset(alldata, diff_dir=diff_dir, fig=None)

    # add the station metadata
    alldata.add_metadata({'station': station.snapshot()})

    alldata.metadata['scanjob'] = scanjob
    alldata.metadata['allgatevalues'] = gvs
    alldata.metadata['scantime'] = str(datetime.datetime.now())
    alldata.metadata['dt'] = dt
    alldata.metadata['wait_time'] = wait_time

    alldata.write(write_metadata=True)

    return alldata


def plotData(alldata, diff_dir=None, fig=1):
    """ Plot a dataset and optionally differentiate """
    plt.figure(fig)
    plt.clf()
    if diff_dir is not None:
        imx = qtt.diffImageSmooth(alldata.measured.ndarray, dy=diff_dir)
        name = 'diff_dir_%s' % diff_dir
        name = uniqueArrayName(alldata, name)
        data_arr = qcodes.DataArray(name=name, label=name, array_id=name, set_arrays=alldata.measured.set_arrays, preset_data=imx)
        alldata.add_array(data_arr)
        plot = MatPlot(interval=0, num=fig)
        plot.add(alldata.arrays[name])
        # plt.axis('image')
        plot.fig.axes[0].autoscale(tight=True)
        plot.fig.axes[1].autoscale(tight=True)
    else:
        plot = MatPlot(interval=0, num=fig)
        plot.add(alldata.default_parameter_array('measured'))
        # plt.axis('image')
        plot.fig.axes[0].autoscale(tight=True)
        try:
            # TODO: make this cleaner code
            plot.fig.axes[1].autoscale(tight=True)
        except:
            pass


#%%


def scan2Dturbo(station, scanjob, sweepgates, sweepranges=[40, 40], verbose=1):
    """Perform a very fast 2d scan by varying two physical gates with the AWG.

    The function assumes the station contains an FPGA with readFPGA function. 
        The FPGA channel is determined form the sensing dot.

    Args:
        station (qcodes.station.Station): contains all data on the measurement station
        scanjob (dict): 

    Returns:
        alldata (qcodes.data.data_set.DataSet): measurement data and metadata
    """
    scanjob['scantype'] = 'scan2Dturbo'
    stepdata = scanjob['stepdata']
    sweepdata = scanjob['sweepdata']

    sweepgates = [stepdata['gate'], sweepdata['gate']]
    sweepranges = [stepdata['end'] - stepdata['start'], sweepdata['end'] - sweepdata['start']]

    sd = scanjob['sd']
    Naverage = scanjob.get('Naverage', 20)
    resolution = scanjob.get('resolution', [90, 90])

    if verbose:
        print('scan2Dturbo: sweepgates %s' % (str(sweepgates),))

    fpga_ch = sd.fpga_ch
    fpga_samp_freq = station.fpga.get_sampling_frequency()

    waveform, sweep_info = station.awg.sweep_2D(fpga_samp_freq, sweepgates, sweepranges, resolution)
    waittime = resolution[0] * resolution[1] * Naverage / fpga_samp_freq

    ReadDevice = ['FPGA_ch%d' % fpga_ch]
    _, DataRead_ch1, DataRead_ch2 = station.fpga.readFPGA(Naverage=Naverage, ReadDevice=ReadDevice, waittime=waittime)

    station.awg.stop()

    dataread = [DataRead_ch1, DataRead_ch2][fpga_ch - 1]
    data = station.awg.sweep_2D_process(dataread, waveform)
    alldata, _ = makeDataset_sweep_2D(data, station.gates, sweepgates, sweepranges)

    alldata.metadata['allgatevalues'] = station.gates.allvalues()
    alldata.metadata['scantime'] = str(datetime.datetime.now())
    alldata.metadata['fpga_samp_freq'] = fpga_samp_freq
    alldata.metadata['scanjob'] = scanjob
    alldata.write()

    return alldata

#%%


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


def scanPinchValue(station, outputdir, gate, basevalues=None, keithleyidx=[1], stepdelay=None, cache=False, verbose=1, fig=10, full=0, background=False):
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
        {'gates': [gate], 'start': max(b, 0), 'end': -750, 'step': -2})
    if full == 0:
        sweepdata['step'] = -6

    scanjob = dict(
        {'sweepdata': sweepdata, 'keithleyidx': keithleyidx, 'wait_time': stepdelay})

    station.gates.set(gate, sweepdata['start'])  # set gate to starting value
    time.sleep(stepdelay)

    alldata = scan1D(scanjob, station, title_comment='scan gate %s' %
                     gate, background=background)

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


def makeDataset_sweep(data, sweepgate, sweeprange, sweepgate_value=None, gates=None, fig=None):
    ''' Convert the data of a 1D sweep to a DataSet

    Note: sweepvalues are only an approximation

    '''
    if sweepgate_value is None:
        if gates is not None:
            sweepgate_param = gates.getattr(sweepgate)
            sweepgate_value = sweepgate_param.get()
        else:
            raise Exception('No gates supplied')

    sweepvalues = np.linspace(sweepgate_value - sweeprange / 2, sweepgate_value + sweeprange / 2, len(data))
    dataset = makeDataSet1Dplain(sweepgate, sweepvalues, yname='measured', y=data)

    if fig is None:
        return dataset, None
    else:
        plot = MatPlot(dataset.measured, interval=0, num=fig)
        return dataset, plot


def makeDataset_sweep_2D(data, gates, sweepgates, sweepranges, fig=None):
    ''' Convert the data of a 2D sweep to a DataSet '''

    gate_horz = getattr(gates, sweepgates[0])
    gate_vert = getattr(gates, sweepgates[1])

    initval_horz = gate_horz.get()
    initval_vert = gate_vert.get()

    sweep_horz = gate_horz[initval_horz - sweepranges[0] /
                           2:sweepranges[0] / 2 + initval_horz:sweepranges[0] / len(data[0])]
    sweep_vert = gate_vert[initval_vert - sweepranges[1] /
                           2:sweepranges[1] / 2 + initval_vert:sweepranges[1] / len(data)]

    dataset = makeDataSet2D(sweep_vert, sweep_horz, preset_data=data)

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
