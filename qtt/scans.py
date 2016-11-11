import qtpy
# print(qtpy.API_NAME)

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
from qtt.algorithms.onedot import onedotGetBalanceFine
import qtt.live

from qtt.data import *

from qcodes.utils.helpers import tprint

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

from qtt.algorithms.generic import *


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

    wait_time = qtt.scans.waitTime(
        od['gates'][2], gate_settle=getattr(station, 'gate_settle', None))

    alldatahi = qtt.scans.scan2D(station, scanjobhi, title_comment='2D scan, local', wait_time=wait_time, background=False)
    extentscan, g0, g2, vstep, vsweep, arrayname = dataset2Dmetadata(
        alldatahi, verbose=0, arrayname=None)
    impixel, tr = dataset2image(alldatahi, mode='pixel')

    #_,_,_, im = get2Ddata(alldatahi)
    ptv, fimg, tmp = onedotGetBalanceFine(
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
"""
def complete(self, delay=1.0, txt=''):
        ''' Block untill dataset had completed '''
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
"""


def getParams(station, keithleyidx):
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
    if 'amplitude' in data.arrays.keys():
        return data.amplitude
    if 'amplitude_0' in data.arrays.keys():
        return data.amplitude_0
    if 'amplitude_1' in data.arrays.keys():
        return data.amplitude_1

    vv = [v for v in (data.arrays.keys()) if v.endswith('amplitude')]
    if (len(vv) > 0):
        name = vv[0]
        return getattr(data, name)

    try:
        name = next(iter(data.arrays.keys()))
        return getattr(data, name)
    except:
        pass
    return None


def scan1D(scanjob, station, location=None, delay=.01, liveplotwindow=None, background=False, title_comment=None, wait_time=None):
    ''' Simple 1D scan '''
    gates = station.gates
    sweepdata = scanjob['sweepdata']
    gate = sweepdata.get('gate', None)
    if gate is None:
        gate = sweepdata.get('gates')[0]
    param = getattr(gates, gate)
    sweepvalues = param[sweepdata['start']:sweepdata['end']:sweepdata['step']]

    if wait_time is not None:
        delay = wait_time
    t0 = time.time()

    # legacy code...
    minstrument = scanjob.get('instrument', None)
    if minstrument is None:
        minstrument = scanjob.get('keithleyidx', None)
    params = getParams(station, minstrument)

    station.set_measurement(*params)

    if background:
        data_manager = None
    else:
        data_manager = False

    delay = scanjob.get('delay', delay)
    if delay is None:
        delay = 0
    logging.debug('delay: %s' % str(delay))
    print('scan1D: starting Loop (background %s)' % background)
    data = qc.Loop(sweepvalues, delay=delay, progress_interval=1).run(
        location=location, data_manager=data_manager, background=background)
    data.sync()

    if liveplotwindow is None:
        liveplotwindow = qtt.live.livePlot()

    if liveplotwindow is not None:
        time.sleep(.1)
        data.sync()  # wait for at least 1 data point
        liveplotwindow.clear()
        liveplotwindow.add(getDefaultParameter(data))

    # FIXME
    if background:
        data.background_functions = dict({'qt': pg.mkQApp().processEvents})
        data.complete(delay=.25)
        data.sync()
        dt = -1
        if qcodes.get_bg() is not None:
            logging.info('background measurement not completed')
            time.sleep(.1)

        logging.info('scan1D: completed %s' % (str(data.location),))
    else:
        dt = time.time() - t0
    if not hasattr(data, 'metadata'):
        data.metadata = dict()

    if 1:
        metadata = data.metadata
        metadata['allgatevalues'] = gates.allvalues()
        metadata['scantime'] = str(datetime.datetime.now())
        metadata['dt'] = dt
        metadata['scanjob'] = scanjob

    logging.info('scan1D: done %s' % (str(data.location),))

    return data

import pyqtgraph as pg


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


def scan2D(station, scanjob, title_comment='', liveplotwindow=None, wait_time=None, background=False):
    """ Make a 2D scan and create dictionary to store on disk

    Args:
        station (object): contains all data on the measurement station
        scanjob (dict): data for scan range
    """

    stepdata = scanjob['stepdata']
    sweepdata = scanjob['sweepdata']
    minstrument = scanjob.get('instrument', None)
    if minstrument is None:
        minstrument = scanjob.get('keithleyidx', None)

    logging.info('scan2D: todo: implement compensategates')
    logging.info('scan2D: todo: implement wait_time')
    # compensateGates = scanjob.get('compensateGates', [])
    # gate_values_corners = scanjob.get('gate_values_corners', [[]])

#    if wait_time == None:
#        wait_time = getwaittime(sweepdata['gates'][0])

    
    delay = scanjob.get('delay', 0.0)
    if wait_time is not None:
        delay = wait_time

    # readdevs = ['keithley%d' % x for x in keithleyidx]

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

    measurement = fullloop.each(*params)

    if background is None:
        try:
            gates._server_name
            background = True
        except:
            background = False

    if background:
        data_manager = None
    else:
        data_manager = False

    alldata = measurement.run(background=background, data_manager=data_manager)

    if liveplotwindow is None:
        liveplotwindow = qtt.live.livePlot()
    if liveplotwindow is not None:
        liveplotwindow.clear()
        liveplotwindow.add(getDefaultParameter(alldata))

    if background is True:
        alldata.background_functions = dict({'qt': pg.mkQApp().processEvents})
        alldata.complete(delay=.5)
        wait_bg_finish()

    dt = time.time() - t0

    if not hasattr(alldata, 'metadata'):
        alldata.metadata = dict()

    alldata.metadata['scanjob'] = scanjob
    alldata.metadata['allgatevalues'] = gates.allvalues()
    alldata.metadata['scantime'] = str(datetime.datetime.now())
    alldata.metadata['dt'] = dt
    alldata.metadata['wait_time'] = wait_time

    if 0:
        # FIXME...
        alldata = copy.copy(scanjob)
        alldata['wait_time'] = wait_time
        alldata['data_array'] = data.get_data()
        alldata['datadir'] = data._dir
        alldata['timemark'] = data._timemark
        alldata['gatevalues'] = gatevalues(activegates)
        alldata['gatevalues']['T'] = get_gate('T')
        # basename='%s-sweep-2d-%s' % (idstr, 'x-%s-%s' % (gg[0], gg[2]) )
        # save(os.path.join(xdir, basename +'.pickle'), alldata )
        
    # save the dataset to disk
    name = '_'.join(sai.array_id for sai in alldata.measured.set_arrays)
    alldata.location = alldata.location_provider(alldata.default_io, record={'name': name})
    alldata.write()


    return alldata

# Causes errors with the IVVI


def scan2Dfast(station, scanjob, liveplotwindow=None, diff_dir=None, wait_time=None, background=None):
    """ Make a 2D scan and create dictionary to store on disk

    Arguments:
        station (object): contains all data on the measurement station
        scanjob (dict): data for scan range
    """

    stepdata = scanjob['stepdata']
    sweepdata = scanjob['sweepdata']
    Naverage = scanjob.get('Naverage', 20)

#    logging.info('scan2D: todo: implement compensategates for sensing dot compensation')

    delay = scanjob.get('delay', 0.0)
    if wait_time is not None:
        raise Exception('not implemented')

    gates = station.gates

    sweepgate = sweepdata.get('gate', None)
    sweepparam = getattr(gates, sweepgate)

    stepgate = stepdata.get('gate', None)
    stepparam = getattr(gates, stepgate)

    def readfunc(waveform, Naverage):
        fpga_ch = scanjob['sd'].fpga_ch
        ReadDevice = ['FPGA_ch%d' % fpga_ch]
        data_raw = np.array(station.fpga.readFPGA(ReadDevice=ReadDevice, Naverage=Naverage)[fpga_ch])
        data = station.awg.sweep_process(data_raw, waveform, Naverage)
        return data

    sweeprange = (sweepdata['end'] - sweepdata['start'])
#    sweeprange = qtt.algorithms.generic.signedmin(sweeprange, 60)  # FIXME
    period = scanjob['sweepdata'].get('period', 1e-3)
    sweepgate_value = (sweepdata['start'] + sweepdata['end']) / 2
    
    
#    if 'gates_vert' in scanjob.keys():
#        steprange = (stepdata['end']-stepdata['start'])
#        for g in scanjob['gates_vert']:
#            gates.set(g,gates.get(g)+scanjob['gates_vert'][g]*steprange/2)
#    else:
#        stepparam.set(stepdata['start'])
    
    if 'gates_vert' in scanjob.keys():
        scanjob['gates_vert_init'] = {}
        for g in scanjob['gates_vert']:
            gates.set(g,gates.get(g)+(stepdata['start']-stepdata['end'])*scanjob['gates_vert'][g]/2)
            scanjob['gates_vert_init'][g] = gates.get(g)
    else:
        stepparam.set(stepdata['start'])
        sweepparam.set(sweepgate_value)
    
    if 'gates_horz' in scanjob.keys():
        waveform, sweep_info = station.awg.sweep_gate_virt(scanjob['gates_horz'], sweeprange, period)
    else:
        waveform, sweep_info = station.awg.sweep_gate(sweepgate, sweeprange, period)

    qtt.time.sleep(2.5)

    data = readfunc(waveform, Naverage)
    ds0, _ = makeDataset_sweep(data, sweepgate, sweeprange, sweepgate_value=sweepgate_value, fig=None)

    sweepvalues = sweepparam[list(ds0.arrays[sweepgate])]
    stepvalues = stepparam[stepdata['start']:stepdata['end']:stepdata['step']]

    logging.info('scan2D: %d %d' % (len(stepvalues), len(sweepvalues)))
    logging.info('scan2D: delay %f' % delay)

    t0 = qtt.time.time()

    if background is None:
        try:
            gates._server_name
            background = True
        except:
            background = False

    alldata = makeDataSet2D(stepvalues, sweepvalues)

    if liveplotwindow is None:
        liveplotwindow = qtt.live.livePlot()
    if liveplotwindow is not None:
        liveplotwindow.clear()
        liveplotwindow.add(alldata.default_parameter_array(paramname='measured'))

    for ix, x in enumerate(stepvalues):
        tprint('scan2Dfast: %d/%d: setting %s to %.3f' % (ix, len(stepvalues), stepvalues.name, x), dt=.5)
        if 'gates_vert' in scanjob.keys():
            for g in scanjob['gates_vert']:
                gates.set(g,scanjob['gates_vert_init'][g]+ix*stepdata['step']*scanjob['gates_vert'][g])
        else:
            stepvalues.set(x)
        qtt.time.sleep(delay)
        alldata.measured.ndarray[ix] = readfunc(waveform, Naverage)
        liveplotwindow.update_plot()
        liveplotwindow.fig.axes[0].autoscale(tight=True)
        liveplotwindow.fig.axes[1].autoscale(tight=True)
        pg.mkQApp().processEvents()

    station.awg.stop()

    dt = qtt.time.time() - t0

    if not hasattr(alldata, 'metadata'):
        alldata.metadata = dict()
        
    if diff_dir is not None:
        imx = qtt.diffImageSmooth(alldata.measured.ndarray, dy=diff_dir)
        name = 'diff_dir_%s' % diff_dir
        data_arr = qcodes.DataArray(name=name, label=name, array_id=name, set_arrays=alldata.measured.set_arrays, preset_data=imx)
        alldata.add_array(data_arr)
        plot = MatPlot(interval=0)
        plot.add(alldata.arrays[name])
        plot.fig.axes[0].autoscale(tight=True)
        plot.fig.axes[1].autoscale(tight=True)

    alldata.metadata['scanjob'] = scanjob
    alldata.metadata['allgatevalues'] = gates.allvalues()
    alldata.metadata['scantime'] = str(datetime.datetime.now())
    alldata.metadata['dt'] = dt
    alldata.metadata['wait_time'] = wait_time

    # save the dataset to disk
    name = '_'.join(sai.array_id for sai in alldata.measured.set_arrays)
    alldata.location = alldata.location_provider(alldata.default_io, record={'name': name})
    alldata.write()

    return alldata

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
        stepdelay = waitTime(gate, gate_settle=getattr(
            station, 'gate_settle', None))

    if basevalues is None:
        b = 0
    else:
        b = basevalues[gate]
    sweepdata = dict(
        {'gates': [gate], 'start': max(b, 0), 'end': -750, 'step': -2})
    if full == 0:
        sweepdata['step'] = -6

    scanjob = dict(
        {'sweepdata': sweepdata, 'keithleyidx': keithleyidx, 'delay': stepdelay})

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
