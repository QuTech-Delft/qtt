""" Basic scan functions

This module contains functions for basic scans, e.g. scan1D, scan2D, etc.
This is part of qtt. 
"""

import datetime
import logging
import os
import re
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph
import qcodes
import skimage
import skimage.filters
from qcodes import DataArray, Instrument
from qcodes.instrument.parameter import Parameter, StandardParameter
from qcodes.plots.qcmatplotlib import MatPlot
from qcodes.utils.helpers import tprint

import qtt.algorithms.onedot
import qtt.gui.live_plotting
import qtt.instrument_drivers.virtualAwg.virtual_awg
import qtt.utilities.tools
from qtt.algorithms.gatesweep import analyseGateSweep
from qtt.instrument_drivers.simulation_instruments import SimulationDigitizer
from qtt.measurements.acquisition.interfaces import AcquisitionScopeInterface
import qtt.instrument_drivers.simulation_instruments

from qtt.data import makeDataSet1D, makeDataSet2D, makeDataSet1Dplain, makeDataSet2Dplain
from qtt.data import diffDataset, loadDataset
from qtt.data import uniqueArrayName

from qtt.utilities.tools import update_dictionary
from qtt.structures import VectorParameter

# %%


def checkReversal(im0, verbose=0):
    """ Check sign of a current scan

    We assume that the current is either zero or positive 
    Needed when the keithley (or some other measurement device) has been reversed

    Args:
        im0 (array): measured data
    Returns
        bool
    """
    thr = skimage.filters.threshold_otsu(im0)
    mval = np.mean(im0)

    # meanopen = np.mean(im0[:,:])
    fr = thr < mval
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


# %%


def instrumentName(namebase):
    """ Return name for qcodes instrument that is available

    Args:
        namebase (str)
    Returns:
        name (str)
    """
    inames = qcodes.Instrument._all_instruments
    name = namebase
    for ii in range(10000):
        if not (name in inames):
            return name
        else:
            name = namebase + '%d' % ii
    raise Exception(
        'could not find unique name for instrument with base %s' % namebase)


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
    sweepdata = dict(
        {'param': g1, 'start': r1[0], 'end': r1[1], 'step': float(step)})
    scanjob = scanjob_t({'sweepdata': sweepdata, 'minstrument': keithleyidx})
    if not g2 is None:
        stepdata = dict(
            {'param': g2, 'start': r2[0], 'end': r2[1], 'step': float(step)})
        scanjob['stepdata'] = stepdata

    return scanjob


# %%

@qtt.utilities.tools.deprecated
def _parse_stepdata(stepdata):
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
    if isinstance(v, (str, StandardParameter, Parameter, dict)):
        pass
    elif isinstance(v, list):
        warnings.warn('please use string or Instrument instead of list')
        stepdata['param'] = stepdata['param'][0]

    if 'range' in stepdata:
        if 'end' in 'stepdata':
            if stepdata['end'] != stepdata['start'] + stepdata['range']:
                warnings.warn(
                    'in scanjob the start, end and range arguments do not match')
        stepdata['end'] = stepdata['start'] + stepdata['range']
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


# %%


@qtt.utilities.tools.rdeprecated(txt='Method will be removed in future release of qtt. Use qtt.data.plot_dataset', expire='1 Sep 2018')
def plot1D(data, fig=100, mstyle='-b'):
    """ Show result of a 1D scan """

    val = data.default_parameter_name()

    if fig is not None:
        plt.figure(fig)
        plt.clf()
        MatPlot(getattr(data, val), interval=None, num=fig)


# %%

def get_instrument_parameter(handle):
    """ Return handle to instrument parameter or channel

    Args:
        handle (tuple or str): name of instrument or handle. Tuple is a pair (instrument, paramname). A string is
        of the form 'instrument.parameter', e.g. 'keithley3.amplitude'
    Returns:
        h (object)
    """
    if isinstance(handle, str):
        try:
            istr, pstr = handle.split('.')
        except Exception as ex:
            # probably legacy code
            istr = handle
            pstr = 'amplitude'
            warnings.warn('incorrect format for instrument+parameter handle %s' % (handle,))

    else:
        istr, pstr = handle

    if isinstance(istr, str):
        instrument = qcodes.Instrument.find_instrument(istr)
    else:
        instrument = istr

    if isinstance(pstr, int):
        pstr = 'channel_%d' % pstr

    param = getattr(instrument, pstr)
    return instrument, param


def get_instrument(instr, station=None):
    """ Return handle to instrument

    Args:
        instr (str, Instrument, tuple, list): name of instrument or handle or pair (handle, channel)
    """

    if isinstance(instr, (tuple, list)):
        # assume the tuple is (instrument, channel)
        instr = instr[0]

    if isinstance(instr, Instrument):
        return instr

    if isinstance(instr, AcquisitionScopeInterface):
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
            ref = station.components[instr]
            return ref
    raise Exception('could not find instrument %s' % str(instr))


def get_minstrument_channels(minstrument):
    if isinstance(minstrument, tuple):
        minstrument = minstrument[1]

    if isinstance(minstrument, int):
        read_ch = [minstrument]
        return read_ch

    if isinstance(minstrument, list):
        read_ch = minstrument
        return read_ch

    raise Exception('could not parse %s into channels' % minstrument)


def get_measurement_params(station, mparams):
    """ Get qcodes parameters from an index or string or parameter """
    params = []
    digi_flag = False
    channels = []
    if isinstance(mparams, (int, str, Parameter, tuple)):
        # for convenience
        mparams = [mparams]
    elif isinstance(mparams, (list, tuple)):
        pass
    else:
        warnings.warn('unknown argument type')
    for x in mparams:
        if isinstance(x, int):
            params += [getattr(station, 'keithley%d' % x).amplitude]
            warnings.warn('please use a string to specify your parameter, e.g. \'keithley3.amplitude\'!')
        elif isinstance(x, tuple):
            instrument, param = get_instrument_parameter(x)
            params += [param]

        elif isinstance(x, str):
            if x.startswith('digitizer'):
                channels.append(int(x[-1]))
                digi_flag = True
                params += [getattr(station.digitizer, 'channel_%c' % x[-1])]
            elif '.' in x:
                params += [get_instrument_parameter(x)[1]]
            else:
                warnings.warn(
                    'legacy style argument, please use \'keithley1.amplitude\' or (keithley1.name, \'amplitude\')')
                params += [getattr(station, x).amplitude]
        else:
            params += [x]
    if digi_flag:
        station.digitizer.initialize_channels(
            channels, memsize=station.digitizer.data_memory_size())
    return params


def getDefaultParameter(data):
    """ Return name of the main array in the dataset """
    return data.default_parameter_name()


# %%

def _add_dataset_metadata(dataset):
    """ Add different kinds of metadata to a dataset """
    update_dictionary(dataset.metadata, scantime=str(datetime.datetime.now()))
    update_dictionary(dataset.metadata, code_version=qtt.utilities.tools.code_version())
    update_dictionary(dataset.metadata, __dataset_metadata=qtt.data.dataset_to_dictionary(
        dataset, include_data=False, include_metadata=False))


def _initialize_live_plotting(alldata, plotparam, liveplotwindow=None, subplots=False):
    """ Initialize live plotting

    Args:
        alldata (DataSet): DataSet to plot from
        plotparam (str or list or None): Arrays in the DataSet to plot. If None then automatically select. If False then perform no live plotting
        liveplotwindow (None or object): handle to live plotting window
    """
    if plotparam is False:
        return None

    if liveplotwindow is None:
        liveplotwindow = qtt.gui.live_plotting.getLivePlotWindow()
    if liveplotwindow:
        liveplotwindow.clear()
        if isinstance(plotparam, (list, tuple)):
            for ii, plot_parameter in enumerate(plotparam):
                if subplots:
                    liveplotwindow.add(alldata.default_parameter_array(paramname=plot_parameter), subplot=ii + 1)
                else:
                    liveplotwindow.add(alldata.default_parameter_array(paramname=plot_parameter))
        elif plotparam is None:
            liveplotwindow.add(alldata.default_parameter_array())
        else:
            liveplotwindow.add(alldata.default_parameter_array(paramname=plotparam))

        pyqtgraph.mkQApp().processEvents()  # needed for the parameterviewer
    return liveplotwindow


def scan1D(station, scanjob, location=None, liveplotwindow=None, plotparam='measured', verbose=1, extra_metadata=None):
    """Simple 1D scan. 

    Args:
        station (object): contains all data on the measurement station
        scanjob (scanjob_t): data for scan
        extra_metadata (None or dict): additional metadata to be included in the dataset

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

    scanjob._parse_stepdata('sweepdata')

    scanjob.parse_param('sweepdata', station, paramtype='slow')

    if isinstance(scanjob['sweepdata']['param'], lin_comb_type):
        scanjob['scantype'] = 'scan1Dvec'
        scanjob._start_end_to_range()
    else:
        scanjob['scantype'] = 'scan1D'

    sweepdata = scanjob['sweepdata']

    sweepvalues = scanjob._convert_scanjob_vec(station)

    wait_time = sweepdata.get('wait_time', 0)
    wait_time_startscan = scanjob.get('wait_time_startscan', 2 * wait_time)
    t0 = time.time()

    logging.debug('wait_time: %s' % str(wait_time))

    alldata, (set_names, measure_names) = makeDataSet1D(sweepvalues, yname=mparams,
                                                        location=location, loc_record={'label': scanjob['scantype']},
                                                        return_names=True)

    liveplotwindow = _initialize_live_plotting(alldata, plotparam, liveplotwindow)

    def myupdate():
        if liveplotwindow:
            t0 = time.time()
            liveplotwindow.update()
            if verbose >= 2:
                print('scan1D: myupdate: %.3f ' % (time.time() - t0))

    tprev = time.time()
    for ix, x in enumerate(sweepvalues):
        if verbose:
            tprint('scan1D: %d/%d: time %.1f' %
                   (ix, len(sweepvalues), time.time() - t0), dt=1.5)

        if scanjob['scantype'] == 'scan1Dvec':
            for param in scanjob['phys_gates_vals']:
                gates.set(param, scanjob['phys_gates_vals'][param][ix])
        else:
            sweepvalues.set(x)
        if ix == 0:
            time.sleep(wait_time_startscan)
        else:
            time.sleep(wait_time)
        for ii, p in enumerate(mparams):
            value = p.get()
            alldata.arrays[measure_names[ii]].ndarray[ix] = value

        delta, tprev, update_plot = _delta_time(tprev, thr=.25)
        if update_plot:
            if liveplotwindow:
                myupdate()
            pyqtgraph.mkQApp().processEvents()  # needed for the parameterviewer

        if qtt.abort_measurements():
            print('  aborting measurement loop')
            break

    myupdate()
    dt = time.time() - t0

    if scanjob['scantype'] is 'scan1Dvec':
        for param in scanjob['phys_gates_vals']:
            parameter = gates.parameters[param]
            arr = DataArray(name=parameter.name, array_id=parameter.name, label=parameter.label, unit=parameter.unit,
                            preset_data=scanjob['phys_gates_vals'][param],
                            set_arrays=(alldata.arrays[sweepvalues.parameter.name],))
            alldata.add_array(arr)

    if not hasattr(alldata, 'metadata'):
        alldata.metadata = dict()

    if extra_metadata is not None:
        update_dictionary(alldata.metadata, **extra_metadata)

    update_dictionary(alldata.metadata, scanjob=dict(scanjob),
                      dt=dt, station=station.snapshot())
    update_dictionary(alldata.metadata, allgatevalues=gatevals)
    _add_dataset_metadata(alldata)

    logging.info('scan1D: done %s' % (str(alldata.location),))

    alldata.write(write_metadata=True)

    return alldata


# %%
def scan1Dfast(station, scanjob, location=None, liveplotwindow=None, delete=True, verbose=1, plotparam=None, extra_metadata=None):
    """ Fast 1D scan. The scan is performed by putting a sawtooth signal on the AWG and measuring with a fast acquisition device.

    Args:
        station (object): contains all data on the measurement station
        scanjob (scanjob_t): data for scan
        extra_metadata (None or dict): additional metadata to be included in the dataset

    Returns:
        DataSet: contains the measurement data and metadata
    """
    gates = station.gates
    gatevals = gates.allvalues()

    if 'sd' in scanjob:
        warnings.warn('sd argument is not supported in scan1Dfast')

    if type(scanjob) is dict:
        warnings.warn('Use the scanjob_t class.', DeprecationWarning)
        scanjob = scanjob_t(scanjob)

    scanjob._parse_stepdata('sweepdata')

    scanjob.parse_param('sweepdata', station, paramtype='fast')
    minstrhandle = get_instrument(scanjob.get('minstrumenthandle', 'digitizer'), station=station)
    virtual_awg = getattr(station, 'virtual_awg', None)

    read_ch = scanjob['minstrument']
    if isinstance(read_ch, tuple):
        read_ch = read_ch[1]

    if isinstance(read_ch, int):
        read_ch = [read_ch]

    if isinstance(read_ch, str):
        raise Exception('for fast scans the minstrument should be a list of channel numbers')

    if isinstance(scanjob['sweepdata']['param'], lin_comb_type):
        scanjob['scantype'] = 'scan1Dfastvec'
        fast_sweep_gates = scanjob['sweepdata']['param'].copy()
        scanjob._start_end_to_range()
    else:
        scanjob['scantype'] = 'scan1Dfast'

    sweepdata = scanjob['sweepdata']
    Naverage = scanjob.get('Naverage', 20)
    period = scanjob['sweepdata'].get('period', 1e-3)
    t0 = time.time()
    wait_time_startscan = scanjob.get('wait_time_startscan', 0)

    if scanjob['scantype'] == 'scan1Dfast':
        if 'range' in sweepdata:
            sweeprange = sweepdata['range']
        else:
            sweeprange = (sweepdata['end'] - sweepdata['start'])
            sweepgate_value = (sweepdata['start'] + sweepdata['end']) / 2
            gates.set(sweepdata['param'], float(sweepgate_value))
        if 'pulsedata' in scanjob:
            waveform, sweep_info = station.awg.sweepandpulse_gate({'gate': sweepdata['param'].name,
                                                                   'sweeprange': sweeprange, 'period': period},
                                                                  scanjob['pulsedata'])
        else:
            if virtual_awg:
                measure_gates = {sweepdata['param']: 1}
                waveform = virtual_awg.sweep_gates(measure_gates, sweeprange, period, do_upload=delete)
                virtual_awg.enable_outputs(list(measure_gates.keys()))
                virtual_awg.run()
            else:
                waveform, sweep_info = station.awg.sweep_gate(sweepdata['param'], sweeprange, period, delete=delete)
    else:
        sweeprange = sweepdata['range']
        if 'pulsedata' in scanjob:
            sg = []
            for g, v in fast_sweep_gates.items():
                if v != 0:
                    sg.append(g)
            if len(sg) > 1:
                raise Exception('AWG pulses does not yet support virtual gates')
            waveform, sweep_info = station.awg.sweepandpulse_gate({'gate': sg[0], 'sweeprange': sweeprange,
                                                                   'period': period}, scanjob['pulsedata'])
        else:
            if virtual_awg:
                sweep_range = sweeprange
                waveform = virtual_awg.sweep_gates(fast_sweep_gates, sweep_range, period, do_upload=delete)
                virtual_awg.enable_outputs(list(fast_sweep_gates.keys()))
                virtual_awg.run()
            else:
                waveform, sweep_info = station.awg.sweep_gate_virt(fast_sweep_gates, sweeprange, period, delete=delete)

    time.sleep(wait_time_startscan)
    data = measuresegment(waveform, Naverage, minstrhandle, read_ch)
    sweepvalues = scanjob._convert_scanjob_vec(station, sweeplength=data[0].shape[0])

    if len(read_ch) == 1:
        measure_names = ['measured']
    else:
        measure_names = ['READOUT_ch%d' % c for c in read_ch]

    alldata = makeDataSet1Dplain(sweepvalues.parameter.name, sweepvalues, measure_names, data,
                                 xunit=qtt.data.determine_parameter_unit(sweepvalues.parameter),
                                 yunit=None,
                                 location=location, loc_record={'label': scanjob['scantype']})
    if virtual_awg:
        virtual_awg.stop()
    else:
        station.awg.stop()

    liveplotwindow = _initialize_live_plotting(alldata, plotparam, liveplotwindow)

    dt = time.time() - t0
    if not hasattr(alldata, 'metadata'):
        alldata.metadata = dict()

    if extra_metadata is not None:
        update_dictionary(alldata.metadata, **extra_metadata)

    update_dictionary(alldata.metadata, scanjob=dict(scanjob), dt=dt, station=station.snapshot())
    update_dictionary(alldata.metadata, allgatevalues=gatevals)
    _add_dataset_metadata(alldata)

    alldata = qtt.utilities.tools.stripDataset(alldata)

    alldata.write(write_metadata=True)
    return alldata


# %%


def makeScanjob(sweepgates, values, sweepranges, resolution):
    """ Create a scanjob from sweep ranges and a centre """
    sj = {}

    nx = len(sweepgates)
    step = sweepranges[0] / resolution[0]
    stepdata = {'gates': [sweepgates[0]], 'start': values[0]
                - sweepranges[0] / 2, 'end': values[0] + sweepranges[0] / 2,
                'step': step}
    sj['stepdata'] = stepdata
    if nx == 2:
        step = sweepranges[1] / resolution[1]
        sweepdata = {'gates': [sweepgates[1]], 'start': values[1]
                     - sweepranges[1] / 2, 'end': values[1] + sweepranges[0] / 2,
                     'step': step}
        sj['sweepdata'] = sweepdata
        sj['wait_time_step'] = 4
    return sj


# %%


class sample_data_t(dict):
    """ Hold all kind of sample specific data

    The structure is that of a dictionary. Typical fields:

        gate_boundaries (dict): dictionary with gate boundaries

    """

    def gate_boundaries(self, gate):
        bnds = self.get('gate_boundaries', {})
        b = bnds.get(gate, (None, None))
        return b

    def restrict_boundaries(self, gate, value):
        bnds = self.get('gate_boundaries', {})
        b = bnds.get(gate, (None, None))
        if b[1] is not None:
            value = min(value, b[1])
        if b[0] is not None:
            value = max(value, b[0])
        return value


def _convert_vectorname_to_parametername(vector_name, extra_string=None):
    parameter_name = re.sub(r'\(.*?\)', '', vector_name)
    if extra_string is not None:
        parameter_name += '_' + extra_string
    return parameter_name


class scanjob_t(dict):
    """ Structure that contains information about a scan 

    A typical scanjob contains the following (optional) fields:

    Fields:
        sweepdata (dict):
        stepdata (dict)
        minstrument (str, Parameter or tuple)
        wait_time_startscan (float):

    The sweepdata and stepdata are structures with the following fields:

        param (str, Parameter or dict): parameter to vary
        start, end, step (float)
        wait_time (float)


    Note: currently the scanjob_t is a thin wrapper around a dict.
    """

    def add_sweep(self, param, start, end, step, **kwargs):
        """ Add sweep to scan job """
        end_value = float(end) if end is not None else end
        sweep = {'param': param, 'start': float(start), 'end': end_value, 'step': step, **kwargs}
        if not 'sweepdata' in self:
            self['sweepdata'] = sweep
        elif 'stepdata' not in self:
            self['stepdata'] = sweep
        else:
            raise Exception('3d scans not implemented')

    def add_minstrument(self, minstrument):
        """ Add measurement instrument to scan job """
        self['minstrument'] = minstrument

    def check_format(self):
        """ Check the format of the scanjob for consistency and legacy style arguments """
        if 'stepvalues' in self:
            warnings.warn('please do not use the stepvalues field any more!')

    def setWaitTimes(self, station, min_time=0):
        """ Set default waiting times based on gate filtering """

        gate_settle = getattr(station, 'gate_settle', None)
        t = .1
        if gate_settle is None:
            t = 0
        for f in ['sweepdata', 'stepdata']:
            if f in self:
                if gate_settle:
                    if isinstance(self[f]['param'], dict):
                        gs = float(np.min([gate_settle(g) for g in self[f]['param']]))
                    else:
                        gs = gate_settle(self[f]['param'])

                    if f == 'stepdata':
                        t = 2.5 * gs
                    else:
                        t = gs
                self[f]['wait_time'] = max(t, min_time)
        self['wait_time_startscan'] = .5 + 2 * t

    def _parse_stepdata(self, field, gates=None):
        """ Helper function for legacy code """
        stepdata = self[field]
        if not isinstance(stepdata, dict):
            raise Exception('%s should be dict structure' % field)

        v = stepdata.get('gates', None)
        if v is not None:
            raise Exception('please use param instead of gates')
        v = stepdata.get('gate', None)
        if v is not None:
            warnings.warn('please use param instead of gates',
                          DeprecationWarning)
            stepdata['param'] = stepdata['gate']

        v = stepdata.get('param', None)
        if isinstance(v, list):
            warnings.warn('please use string or Instrument instead of list')
            stepdata['param'] = stepdata['param'][0]
        elif isinstance(v, str):
            if gates is not None:
                stepdata['param'] = getattr(gates, v)
            else:
                pass
        elif isinstance(v, (StandardParameter, Parameter, dict)):
            pass
        self[field] = stepdata

    def parse_param(self, field, station, paramtype='slow'):
        """ Process str params for virtual gates """
        param = self[field]['param']
        if isinstance(param, str):
            virt = None
            if paramtype == 'slow':
                if hasattr(station, 'virts'):
                    virt = station.virts
                elif not hasattr(station, 'gates'):
                    raise Exception(
                        'None of the supported gate instruments were found')
            elif paramtype == 'fast':
                if hasattr(station, 'virtf'):
                    virt = station.virtf
                elif not hasattr(station, 'gates'):
                    raise Exception(
                        'None of the supported gate instruments were found')
            else:
                raise Exception('paramtype must be slow or fast')

            if virt is not None:
                if hasattr(virt, param):
                    param_map = virt.convert_matrix_to_map(
                        virt.get_crosscap_matrix_inv().T, virt._gates_list, virt._virts_list)
                    if 'paramname' not in self[field]:
                        self[field]['paramname'] = param
                    self[field]['param'] = param_map[param]
                elif not hasattr(station.gates, param):
                    raise Exception('unrecognized gate parameter')
            elif not hasattr(station.gates, param):
                raise Exception('unrecognized gate parameter')
        elif isinstance(param, qcodes.instrument.parameter.Parameter):
            self[field]['paramname'] = param.name
        else:
            if 'paramname' not in self[field]:
                def fmt(val):
                    if isinstance(val, float):
                        s = '%.4g' % val
                        if not '.' in s:
                            s += '.'
                        return s
                    else:
                        return str(val)

                self[field]['paramname'] = '_'.join(
                    ['%s(%s)' % (key, fmt(value)) for (key, value) in param.items()])

    def _start_end_to_range(self, scanfields=['stepdata', 'sweepdata']):
        """ Add range to stepdata and/or sweepdata in scanjob.

        Note: This function also converts the start and end fields.        
        """
        if isinstance(scanfields, str):
            scanfields = [scanfields]

        for scanfield in scanfields:
            if scanfield in self:
                scaninfo = self[scanfield]
                if 'range' not in scaninfo:
                    scaninfo['range'] = scaninfo['end'] - scaninfo['start']
                    warnings.warn(
                        'Start and end are converted to a range to scan around the current dc values.')
                    scaninfo['start'] = -scaninfo['range'] / 2
                    scaninfo['end'] = scaninfo['range'] / 2
                else:
                    scaninfo['start'] = -scaninfo['range'] / 2
                    scaninfo['end'] = scaninfo['range'] / 2

    def _parse_2Dvec(self):
        """ Adjust the parameter field in the step- and sweepdata for 2D vector scans.

        This adds coefficient zero for parameters in either the sweep- 
        or the step-parameters that do not exist in the other.

        """
        stepdata = self['stepdata']
        sweepdata = self['sweepdata']
        params = set()
        if isinstance(stepdata['param'], qcodes.Parameter):
            pass
        else:
            vec_check = [(stepdata, isinstance(stepdata['param'], lin_comb_type)),
                         (sweepdata, isinstance(sweepdata['param'], lin_comb_type))]
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

    def _convert_scanjob_vec(self, station, sweeplength=None, steplength=None, stepvalues=None):
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
            if 'range' in sweepdata:
                if self['scantype'] in ['scan1Dvec', 'scan1Dfastvec']:
                    sweepdata['start'] = -sweepdata['range'] / 2
                    sweepdata['end'] = sweepdata['range'] / 2
                else:
                    param_val = gates.get(sweepdata['param'])
                    sweepdata['start'] = param_val - sweepdata['range'] / 2
                    sweepdata['end'] = param_val + sweepdata['range'] / 2
            if self['scantype'] in ['scan1Dvec', 'scan1Dfastvec']:
                if 'paramname' in self['sweepdata']:
                    sweepname = self['sweepdata']['paramname']
                else:
                    sweepname = 'sweepparam'
                sweepname_identifier = _convert_vectorname_to_parametername(sweepname, 'sweep_parameter')
                sweepparam = VectorParameter(name=sweepname_identifier, label=sweepname, comb_map=[(
                    gates.parameters[x], sweepdata['param'][x]) for x in sweepdata['param']])
            elif self['scantype'] in ['scan1D', 'scan1Dfast']:
                sweepparam = get_param(gates, sweepdata['param'])
            else:
                raise Exception('unknown scantype')
            if sweeplength is not None:
                sweepdata['step'] = (sweepdata['end']
                                     - sweepdata['start']) / sweeplength
            if self['scantype'] in ['scan1Dvec', 'scan1Dfastvec']:
                last = sweepdata['start'] + sweepdata['range']
                scanvalues = sweepparam[sweepdata['start']:last:sweepdata.get('step', 1.)]

                param_init = {param: gates.get(param)
                              for param in sweepdata['param']}
                self['phys_gates_vals'] = {param: np.zeros(
                    len(scanvalues)) for param in sweepdata['param']}
                sweep_array = np.linspace(-sweepdata['range'] / 2,
                                          sweepdata['range'] / 2, len(scanvalues))
                for param in sweepdata['param']:
                    self['phys_gates_vals'][param] = param_init[param] + \
                        sweep_array * sweepdata['param'][param]
            else:
                scanvalues = sweepparam[sweepdata['start']:sweepdata['end']:sweepdata['step']]

            self['sweepdata'] = sweepdata
        elif self['scantype'][:6] == 'scan2D':
            stepdata = self['stepdata']
            if 'range' in stepdata:
                if self['scantype'] in ['scan2Dvec', 'scan2Dfastvec', 'scan2Dturbovec']:
                    stepdata['start'] = -stepdata['range'] / 2
                    stepdata['end'] = stepdata['range'] / 2
                else:
                    param_val = stepdata['param'].get()
                    stepdata['start'] = param_val - stepdata['range'] / 2
                    stepdata['end'] = param_val + stepdata['range'] / 2
            sweepdata = self['sweepdata']
            if 'range' in sweepdata:
                if self['scantype'] in ['scan2Dvec', 'scan2Dfastvec', 'scan2Dturbovec']:
                    sweepdata['start'] = -sweepdata['range'] / 2
                    sweepdata['end'] = sweepdata['range'] / 2
                else:
                    param_val = sweepdata['param'].get()
                    sweepdata['start'] = param_val - sweepdata['range'] / 2
                    sweepdata['end'] = param_val + sweepdata['range'] / 2
            if self['scantype'] in ['scan2Dvec', 'scan2Dfastvec', 'scan2Dturbovec']:
                if 'paramname' in self['stepdata']:
                    stepname = self['stepdata']['paramname']
                else:
                    stepname = 'stepparam'
                if 'paramname' in self['sweepdata']:
                    sweepname = self['sweepdata']['paramname']
                else:
                    sweepname = 'sweepparam'
                if stepvalues is None:
                    stepname_identifier = _convert_vectorname_to_parametername(stepname, 'step_parameter')
                    stepparam = VectorParameter(name=stepname_identifier, comb_map=[(
                        gates.parameters[x], stepdata['param'][x]) for x in stepdata['param']])
                else:
                    stepparam = self['stepdata']['param']
                sweepname_identifier = _convert_vectorname_to_parametername(sweepname, 'sweep_parameter')
                sweepparam = VectorParameter(name=sweepname_identifier, comb_map=[(
                    gates.parameters[x], sweepdata['param'][x]) for x in sweepdata['param']])
            elif self['scantype'] in ['scan2D', 'scan2Dfast', 'scan2Dturbo']:
                stepparam = stepdata['param']
                sweepparam = sweepdata['param']
            else:
                raise Exception('unknown scantype')
            if sweeplength is not None:
                if 'range' in sweepdata:
                    sweepdata['step'] = sweepdata['range'] / sweeplength
                else:
                    sweepdata['step'] = (
                        sweepdata['end'] - sweepdata['start']) / sweeplength
            if steplength is not None:
                if 'range' in stepdata:
                    stepdata['step'] = stepdata['range'] / steplength
                else:
                    stepdata['step'] = (
                        stepdata['end'] - stepdata['start']) / steplength

            sweepvalues = sweepparam[sweepdata['start']:sweepdata['end']:sweepdata['step']]
            if stepvalues is None:
                stepvalues = stepparam[stepdata['start']:stepdata['end']:stepdata['step']]
            scanvalues = [stepvalues, sweepvalues]
            if self['scantype'] in ['scan2Dvec', 'scan2Dfastvec', 'scan2Dturbovec']:
                param_init = {param: gates.get(param)
                              for param in sweepdata['param']}
                self['phys_gates_vals'] = {param: np.zeros(
                    (len(stepvalues), len(sweepvalues))) for param in sweepdata['param']}
                step_array2d = np.tile(
                    np.array(stepvalues).reshape(-1, 1), (1, len(sweepvalues)))
                sweep_array2d = np.tile(sweepvalues, (len(stepvalues), 1))
                for param in sweepdata['param']:
                    if isinstance(stepvalues, np.ndarray):
                        self['phys_gates_vals'][param] = param_init[param] + sweep_array2d * \
                            sweepdata['param'][param]
                    else:
                        self['phys_gates_vals'][param] = param_init[param] + step_array2d * \
                            stepdata['param'][param] + sweep_array2d * \
                            sweepdata['param'][param]
            self['stepdata'] = stepdata
            self['sweepdata'] = sweepdata

        return scanvalues


def _delta_time(tprev, thr=2):
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


def awgGate(gate, station):
    """ Return True if the specified gate can be controlled by the AWG """
    awg = getattr(station, 'awg', None)
    if awg is None:
        return False
    return awg.awg_gate(gate)


def fastScan(scanjob, station):
    """ Returns whether we can do a fast scan using an awg 

    Args:
        scanjob

    Returns
        f (int): 0: no fast scan possible, 1: scan2Dfast, 2: all axis

    """

    awg = getattr(station, 'awg', None)
    if awg is None:
        awg = getattr(station, 'virtual_awg', None)

    if 'awg' is None:
        return 0

    if isinstance(awg, qtt.instrument_drivers.virtualAwg.virtual_awg.VirtualAwg):
        awg_map = awg._settings.awg_map
        if isinstance(scanjob['sweepdata']['param'], dict):
            params = scanjob['sweepdata']['param'].keys()
        else:
            params = [scanjob['sweepdata']['param']]
        if not np.all([(param in awg_map) for param in params]):
            # sweep gate is not fast, so no fast scan possible
            return 0
        if 'stepdata' in scanjob:
            if scanjob['stepdata'].get('param', None) in awg_map:
                return 2
        return 1

    warnings.warn('old virtual awg object, ', DeprecationWarning)
    if not awg.awg_gate(scanjob['sweepdata']['param']):
        # sweep gate is not fast, so no fast scan possible
        return 0
    if 'stepdata' in scanjob:
        if awg.awg_gate(scanjob['stepdata'].get('param', None)):
            return 2
    return 1


lin_comb_type = dict
""" Class to represent linear combinations of parameters  """


def scan2D(station, scanjob, location=None, liveplotwindow=None, plotparam='measured', diff_dir=None, write_period=None,
           update_period=5, verbose=1, extra_metadata=None):
    """Make a 2D scan and create dictionary to store on disk.

    For 2D vector scans see also the documentation of the _convert_scanjob_vec
    method of the scanjob_t class.

    Args:
        station (object): contains all the instruments
        scanjob (scanjob_t): data for scan
        write_period (float): save-to-disk interval in lines, None for no writing before finished
        update_period (float): liveplot update interval in lines, None for no updates
        extra_metadata (None or dict): additional metadata to be included in the dataset

    Returns:
        alldata (DataSet): contains the measurement data and metadata
    """
    gates = station.gates
    gatevals = gates.allvalues()

    scanjob.check_format()

    minstrument = parse_minstrument(scanjob)
    mparams = get_measurement_params(station, minstrument)

    if type(scanjob) is dict:
        warnings.warn('Use the scanjob_t class.', DeprecationWarning)
        scanjob = scanjob_t(scanjob)

    scanjob._parse_stepdata('stepdata', gates)
    scanjob._parse_stepdata('sweepdata', gates)

    scanjob.parse_param('sweepdata', station, paramtype='slow')
    scanjob.parse_param('stepdata', station, paramtype='slow')

    if isinstance(scanjob['stepdata']['param'], lin_comb_type) or isinstance(scanjob['sweepdata']['param'],
                                                                             lin_comb_type):
        scanjob['scantype'] = 'scan2Dvec'
        if 'stepvalues' in scanjob:
            scanjob._start_end_to_range(scanfields=['sweepdata'])
        else:
            scanjob._start_end_to_range()
        scanjob._parse_2Dvec()
    else:
        scanjob['scantype'] = 'scan2D'

    scanvalues = scanjob._convert_scanjob_vec(
        station, stepvalues=scanjob.get('stepvalues', None))
    stepvalues = scanvalues[0]
    sweepvalues = scanvalues[1]

    stepdata = scanjob['stepdata']
    sweepdata = scanjob['sweepdata']

    wait_time_sweep = sweepdata.get('wait_time', 0)
    wait_time_step = stepdata.get('wait_time', 0)
    wait_time_startscan = scanjob.get('wait_time_startscan', wait_time_step)
    logging.info('scan2D: %d %d' % (len(stepvalues), len(sweepvalues)))
    logging.info('scan2D: wait_time_sweep %f' % wait_time_sweep)
    logging.info('scan2D: wait_time_step %f' % wait_time_step)

    if type(stepvalues) is np.ndarray:
        stepvalues = stepdata['param'][list(stepvalues[:, 0])]

    alldata, (set_names, measure_names) = makeDataSet2D(stepvalues, sweepvalues,
                                                        measure_names=mparams, location=location, loc_record={
                                                            'label': scanjob['scantype']},
                                                        return_names=True)

    if verbose >= 2:
        print('scan2D: created dataset')
        print('  set_names: %s ' % (set_names,))
        print('  measure_names: %s ' % (measure_names,))

    if plotparam == 'all':
        liveplotwindow = _initialize_live_plotting(alldata, measure_names, liveplotwindow, subplots=True)
    else:
        liveplotwindow = _initialize_live_plotting(alldata, plotparam, liveplotwindow, subplots=True)

    t0 = time.time()
    tprev = time.time()

    # disable time-based write period
    alldata.write_period = None

    for ix, x in enumerate(stepvalues):
        alldata.store((ix,), {stepvalues.parameter.name: x})

        if verbose:
            t1 = time.time() - t0
            t1_str = time.strftime('%H:%M:%S', time.gmtime(t1))
            if (ix == 0):
                time_est = len(sweepvalues) * len(stepvalues) * \
                    scanjob['sweepdata'].get('wait_time', 0) * 2
            else:
                time_est = (t1) / ix * len(stepvalues) - t1
            time_est_str = time.strftime(
                '%H:%M:%S', time.gmtime(time_est))
            if type(stepvalues) is np.ndarray:
                tprint('scan2D: %d/%d, time %s (~%s remaining): setting %s to %s' %
                       (ix, len(stepvalues), t1_str, time_est_str, stepdata['param'].name, str(x)), dt=1.5)
            else:
                tprint('scan2D: %d/%d: time %s (~%s remaining): setting %s to %.3f' %
                       (ix, len(stepvalues), t1_str, time_est_str, stepvalues.name, x), dt=1.5)

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
                    time.sleep(wait_time_startscan)
                else:
                    time.sleep(wait_time_step)
            if wait_time_sweep > 0:
                time.sleep(wait_time_sweep)

            datapoint = {}
            datapoint[sweepvalues.parameter.name] = y

            for ii, p in enumerate(mparams):
                datapoint[measure_names[ii]] = p.get()

            alldata.store((ix, iy), datapoint)

        if write_period is not None:
            if ix % write_period == write_period - 1:
                alldata.write()
                alldata.last_write = time.time()
        if update_period is not None:
            if ix % update_period == update_period - 1:
                delta, tprev, update = _delta_time(tprev, thr=0.5)

                if update and liveplotwindow:
                    liveplotwindow.update_plot()
                    pyqtgraph.mkQApp().processEvents()

        if qtt.abort_measurements():
            print('  aborting measurement loop')
            break
    dt = time.time() - t0

    if liveplotwindow:
        liveplotwindow.update_plot()

    if diff_dir is not None:
        alldata = diffDataset(alldata, diff_dir=diff_dir, fig=None)

    if scanjob['scantype'] == 'scan2Dvec':
        for param in scanjob['phys_gates_vals']:
            parameter = gates.parameters[param]
            if parameter.name in alldata.arrays.keys():
                warnings.warn('parameter %s already in dataset, skipping!' % parameter.name)
                continue

            arr = DataArray(name=parameter.name, array_id=parameter.name, label=parameter.label, unit=parameter.unit, preset_data=scanjob['phys_gates_vals'][param], set_arrays=(
                alldata.arrays[stepvalues.parameter.name], alldata.arrays[sweepvalues.parameter.name]))

            alldata.add_array(arr)

    if not hasattr(alldata, 'metadata'):
        alldata.metadata = dict()

    if extra_metadata is not None:
        update_dictionary(alldata.metadata, **extra_metadata)

    update_dictionary(alldata.metadata, scanjob=dict(scanjob),
                      dt=dt, station=station.snapshot())
    update_dictionary(alldata.metadata, allgatevalues=gatevals)
    _add_dataset_metadata(alldata)

    alldata.write(write_metadata=True)

    return alldata


# %%

def get_sampling_frequency(instrument_handle):
    """ Return sampling frequency of acquisition device

    Args:
        instrument_handle (str or Instrument): handle to instrument
    Returns:
        float: sampling frequency
    """
    instrument_handle = get_instrument(instrument_handle)

    if isinstance(instrument_handle, AcquisitionScopeInterface):
        return instrument_handle.sample_rate
    try:
        import qcodes.instrument_drivers.Spectrum.M4i
        if isinstance(instrument_handle, qcodes.instrument_drivers.Spectrum.M4i.M4i):
            return instrument_handle.sample_rate()
    except ImportError:
        pass
    try:
        import qcodes.instrument_drivers.ZI.ZIUHFLI
        if isinstance(instrument_handle, qcodes.instrument_drivers.ZI.ZIUHFLI.ZIUHFLI):
             return NotImplementedError('not implemented yet')
    except ImportError:
        pass

    if isinstance(instrument_handle, qtt.instrument_drivers.simulation_instruments.SimulationDigitizer):
        return instrument_handle.sample_rate()

    raise Exception('Unrecognized fast readout instrument %s' % instrument_handle)


def process_2d_sawtooth(data, period, samplerate, resolution, width, verbose=0, start_zero=True, fig=None):
    """ Extract a 2D image from a double sawtooth signal

    Args:
        data (numpy array): measured trace
        period (float): period of the full signal
        samplerate (float): sample rate of the acquisition device
        resolution (list): resolution nx, ny. The nx corresonds to the fast oscillating sawtooth
        width (list of float): width paramter of the sawtooth signals
        verbose (int): verbosity level
        start_zero (bool): Default is True
        fig (int or None): figure handle

    Returns

        processed_data (list of arrays): the extracted 2D arrays
        results (dict): contains metadata

    """
    npoints_expected = int(period * samplerate)  # expected number of points
    npoints = data.shape[0]
    nchannels = data.shape[1]
    period_x = period / (resolution[1])
    period_y = period

    if verbose:
        print('process_2d_sawtooth: expected %d data points, got %d' % (npoints_expected, npoints, ))

    if np.abs(npoints - npoints_expected) > 0:
        raise Exception('process_2d_sawtooth: expected %d data points, got %d' % (npoints_expected, npoints, ))

    full_trace = False
    if start_zero and (not full_trace):
        padding_x_time = ((1 - width[0]) / 2) * period_x
        padding_y_time = ((1 - width[1]) / 2) * period_y

        sawtooth_centre_pixels = ((1 - width[1]) / 2 + .5 * width[1]) * period * samplerate
        start_forward_slope_step_pixels = ((1 - width[1]) / 2) * period * samplerate
        end_forward_slope_step_pixels = ((1 - width[1]) / 2 + width[1]) * period * samplerate

    else:
        if full_trace == True:
            padding_x_time = 0
            padding_y_time = 0
            sawtooth_centre_pixels = .5 * width[1] * period * samplerate
        else:
            padding_x_time = 0
            padding_y_time = 0
            sawtooth_centre_pixels = .5 * width[1] * period * samplerate
            start_forward_slope_step_pixels = 0
            end_forward_slope_step_pixels = (width[1]) * period * samplerate

    padding_x = int(padding_x_time * samplerate)
    padding_y = int(padding_y_time * samplerate)

    width_horz = width[0]
    width_vert = width[1]
    res_horz = int(resolution[0])
    res_vert = int(resolution[1])

    if resolution[0] % 32 != 0 or resolution[1] % 32 != 0:
        # send out warning, due to rounding of the digitizer memory buffers
        # this is not supported
        raise Exception(
            'resolution for digitizer is not a multiple of 32 (%s) ' % (resolution,))
    if full_trace:
        npoints_forward_x = int(res_horz)
        npoints_forward_y = int(res_vert)
    else:
        npoints_forward_x = int(width_horz * res_horz)
        npoints_forward_y = int(width_vert * res_vert)

    if verbose:
        print('process_2d_sawtooth: number of points in forward trace (horizontal) %d, vertical %d' %
              (npoints_forward_x, npoints_forward_y, ))
        print('   horizontal mismatch %d/%.1f' % (npoints_forward_x, width_horz * period_x * samplerate))

    processed_data = []
    row_offsets = res_horz * np.arange(0, npoints_forward_y).astype(int) + int(padding_y) + int(padding_x)
    for channel in range(nchannels):
        row_slices = [data[(idx):(idx + npoints_forward_x), channel] for idx in row_offsets]
        processed_data.append(np.array(row_slices))

    if verbose:
        print('process_2d_sawtooth: processed_data shapes: %s' % ([array.shape for array in processed_data]))

    if fig is not None:
        pixel_to_axis = 1. / samplerate
        times = np.arange(npoints) / samplerate
        plt.figure(fig)
        plt.clf()

        plt.plot(times, data[:, :], '.-', label='raw data')
        plt.title('Processing of digitizer trace')
        plt.axis('tight')

        for row_offset in row_offsets:
            qtt.pgeometry.plot2Dline([-1, 0, pixel_to_axis * row_offset, ], ':', color='r', linewidth=.8, alpha=.5)

        qtt.pgeometry.plot2Dline([-1, 0, pixel_to_axis * sawtooth_centre_pixels], '-c',
                                 linewidth=1, label='centre of sawtooth', zorder=-10)
        qtt.pgeometry.plot2Dline([0, -1, 0, ], '-', color=(0, 1, 0, .41), linewidth=.8)

        qtt.pgeometry.plot2Dline([-1, 0, pixel_to_axis * start_forward_slope_step_pixels],
                                 ':k', label='start of step forward slope')
        qtt.pgeometry.plot2Dline([-1, 0, pixel_to_axis * end_forward_slope_step_pixels],
                                 ':k', label='end of step forward slope')

        qtt.pgeometry.plot2Dline([-1, 0, 0, ], '-', color=(0, 1, 0, .41), linewidth=.8, label='start trace')
        qtt.pgeometry.plot2Dline([-1, 0, period, ], '-', color=(0, 1, 0, .41), linewidth=.8, label='end trace')

        #qtt.pgeometry.plot2Dline([0, -1, data[0,3], ], '--', color=(1, 0, 0, .41), linewidth=.8, label='first value of data')

        plt.legend(numpoints=1)

        if verbose >= 2:
            plt.figure(fig + 10)
            plt.clf()
            plt.plot(row_slices[0], '.-r', label='first trace')
            plt.plot(row_slices[-1], '.-b', label='last trace')
            plt.plot(row_slices[int(len(row_slices) / 2)], '.-c')
            plt.legend()

    return processed_data, {'row_offsets': row_offsets, 'period': period}


def process_1d_sawtooth(data, width, period, samplerate, resolution=None, padding=0, start_zero=False,
                        fig=None, verbose=0):
    """ Process data from the M4i and a sawtooth trace 

    This is done to remove the extra padded data of the digitizer and to 
    extract the forward trace of the sawtooth.

    Args:
        data (Nxk array)
        width (float): width of the sawtooth
        period (float)
        samplerate (float): sample rate of digitizer
    Returns
        processed_data (Nxk array): processed data
        rr (tuple)
    """
    npoints_expected = int(period * samplerate)  # expected number of points
    npoints = data.shape[0]
    if verbose:
        print('process_1d_sawtooth: expected %d data points, got %d' % (npoints_expected, npoints, ))

    if len(width) != 1:
        raise Exception('specification is not for a 1D sawtooth signal')
    npoints2 = int(width[0] * period * samplerate)
    npoints2 = npoints2 - (npoints2 % 2)
    trace_start = int(padding)
    trace_end = trace_start + int(npoints2)
    if start_zero:
        delta = int(period * samplerate * (1 - width[0]) / 2)
        trace_start += delta
        trace_end += delta
    processed_data = data[trace_start:trace_end, :].T

    if verbose:
        print('process_1d_sawtooth: processing data: shape %s: trace_start %s, trace_end %s' %
              (data.shape, trace_start, trace_end))

    cc = (trace_start + trace_end) / 2
    if fig is not None:
        plt.figure(fig)
        plt.clf()
        plt.plot(data, label='raw data')
        plt.title('Processing of digitizer trace')
        plt.axis('tight')

        qtt.pgeometry.plot2Dline([-1, 0, cc], '-c', linewidth=1, label='centre of sawtooth', zorder=-10)
        qtt.pgeometry.plot2Dline([0, -1, 0, ], '-', color=(0, 1, 0, .41), linewidth=.8)
        qtt.pgeometry.plot2Dline([-1, 0, trace_start], ':k', label='start of forward slope')
        qtt.pgeometry.plot2Dline([-1, 0, trace_end], ':k', label='end of forward slope')

        plt.legend(numpoints=1)

    return processed_data, (trace_start, trace_end)

def ceilN(x, n):
    return int(n*np.ceil(x/n))
def floorN(x, n):
    return int(n*np.floor(x//n))

def select_m4i_memsize(digitizer, period, trigger_delay=None, nsegments=1, verbose=1, trigger_re_arm_compensation = False):
    """ Select suitable memory size for a given period

    The selected memory size is the period times the sample rate, but rounded above to a multiple of 16.
    Additionally, extra pixels are added because of pretrigger_memsize requirements of the m4i.

    Args:
        digitizer (object)
        period (float): period of signal to measure
        trigger_delay (float): delay in seconds between ingoing signal and returning signal
        nsegments (int): number of segments of period length to fit in memory
        trigger_arm_compensation (bool): In block average mode the M4i needs a time of 40 samples + pretrigger to 
            re-arm the triggering. With this option the segment size is reduced. The signal_end can be larger then
            the segment size.
    Returns:
        memsize (int): total memory size selected
        pre_trigger (int): size of pretrigger selected
        signal_start (int): starting position of signal in pixels
        signal_end (int): end position of signal in pixels

    """
    sample_rate = digitizer.exact_sample_rate()
    if sample_rate == 0:
        raise Exception('digitizer samplerate is zero, please reset digitizer')
    number_points_period = int(period * sample_rate)

    if trigger_delay is None or trigger_delay == 0:
        trigger_delay = 0
    else:
        warnings.warn('non-zero trigger_delay is untested')
    trigger_delay_points = 16 * trigger_delay

    basic_pretrigger_size = 16
    base_segment_size = ceilN(number_points_period + trigger_delay_points, 16) + basic_pretrigger_size

    memsize = base_segment_size * nsegments
    if memsize > digitizer.memory():
        raise Exception(f'Trying to acquire too many points. Reduce sampling rate, period {period} or number segments {nsegments}')

    pre_trigger = ceilN(trigger_delay * sample_rate, 16) + basic_pretrigger_size
    post_trigger = ceilN(base_segment_size - pre_trigger, 16)

    if trigger_re_arm_compensation:
        max_segment_size_re_arm = floorN(number_points_period - pre_trigger - 40, 16)
        if verbose:
            print(f'select_m4i_memsize: post_trigger {post_trigger}, max_segment_size_rearm {max_segment_size_re_arm}')
        post_trigger = min(post_trigger, max_segment_size_re_arm)
        memsize = (pre_trigger + post_trigger)*nsegments

    signal_start = basic_pretrigger_size + int(trigger_delay_points)
    signal_end = signal_start + number_points_period

    digitizer.data_memory_size.set(memsize)
    digitizer.posttrigger_memory_size(post_trigger)
    if verbose:
        print('select_m4i_memsize %s: sample rate %.3f Mhz, period %f [ms]' % (
            digitizer.name, sample_rate / 1e6, period * 1e3))
        print('select_m4i_memsize %s: trace %d points, selected memsize %d' %
              (digitizer.name, number_points_period, memsize))
        print('select_m4i_memsize %s: pre and post trigger: %d %d' % (digitizer.name,
                                                                      digitizer.pretrigger_memory_size(),
                                                                      digitizer.posttrigger_memory_size()))
        print('select_m4i_memsize %s: signal_start %d, signal_end %d' % (digitizer.name, signal_start, signal_end))
    return memsize, pre_trigger, signal_start, signal_end

def _trigger_re_arm_padding(data, number_of_samples, verbose=0):
    """ Pad array to specified size in last dimension """
    re_arm_padding = (number_of_samples)-data.shape[-1]
    data = np.concatenate( (data, np.zeros( data.shape[:-1]+(re_arm_padding, ))), axis=-1 )
    if verbose:
        print(f'measure_raw_segment_m4i: re-arm padding: {re_arm_padding}')        
    return data

def measure_raw_segment_m4i(digitizer, period, read_ch, mV_range, Naverage=100, verbose=0, trigger_re_arm_compensation = False, trigger_re_arm_padding = True):
    """ Record a trace from the digitizer

    Args:
        digitizer (obj): handle to instrument
        period (float): length of segment to read
        read_ch (list): channels to read from the instrument
        mV_range (float): range for input
        Naverage (int): number of averages to perform
        verbose (int): verbosity level
        trigger_arm_compensation (bool): In block average mode the M4i needs a time of 40 samples + pretrigger to 
            re-arm the triggering. With this option this is compensated for by measuring less samples and padding with zeros.
        trigger_re_arm_padding (bool): If True then remove any samples from the trigger re-arm compensation with zeros.

    """
    sample_rate = digitizer.exact_sample_rate()
    maxrate = digitizer.max_sample_rate()
    if sample_rate == 0:
        raise Exception(
            'sample rate of m4i is zero, please reset the digitizer')
    if sample_rate > maxrate:
        raise Exception(
            'sample rate of m4i is > %d MHz, this is not supported' % (maxrate // 1e6))

    # code for compensating for trigger delays in software
    signal_delay = getattr(digitizer, 'signal_delay', None)

    memsize, _, signal_start, signal_end = select_m4i_memsize(
        digitizer, period, trigger_delay=signal_delay, nsegments=1, verbose=verbose, trigger_re_arm_compensation = trigger_re_arm_compensation)
    post_trigger = digitizer.posttrigger_memory_size()

    if mV_range is None:
        mV_range = digitizer.range_channel_0()

    digitizer.initialize_channels(read_ch, mV_range=mV_range, memsize=memsize, termination=None)
    dataraw = digitizer.blockavg_hardware_trigger_acquisition(
        mV_range=mV_range, nr_averages=Naverage, post_trigger=post_trigger)

    if isinstance(dataraw, tuple):
        dataraw = dataraw[0]
    data = np.transpose(np.reshape(dataraw, [-1, len(read_ch)]))
    data = data[:, signal_start:signal_end]
    if trigger_re_arm_compensation and trigger_re_arm_padding:
        data = _trigger_re_arm_padding(data, signal_end-signal_start, verbose)
    return data

@qtt.utilities.tools.deprecated
def select_digitizer_memsize(digitizer, period, trigger_delay=None, nsegments=1, verbose=1):
    """ Select suitable memory size for a given period

    Args:
        digitizer (object): handle to instrument
        period (float): period of signal to measure
        trigger_delay (float): delay in seconds between ingoing signal and returning signal
        nsegments (int): number of segments of period length to fit in memory
    Returns:
        memsize (int)
    """
    drate = digitizer.sample_rate()
    if drate == 0:
        raise Exception('digitizer samplerate is zero, please reset digitizer')
    npoints = int(period * drate)
    segsize = int(np.ceil(npoints / 16) * 16)
    memsize = segsize * nsegments
    if memsize > digitizer.memory():
        raise (Exception('Trying to acquire too many points. Reduce sampling rate, period or number segments'))
    digitizer.data_memory_size.set(memsize)
    if trigger_delay is None:
        spare = np.ceil((segsize - npoints) / 16) * 16
        pre_trigger = min(spare / 2, 512)
    else:
        pre_trigger = trigger_delay * drate
    post_trigger = int(np.ceil((segsize - pre_trigger) // 16) * 16)
    digitizer.posttrigger_memory_size(post_trigger)
    if verbose:
        print('%s: sample rate %.3f Mhz, period %f [ms]' % (
            digitizer.name, drate / 1e6, period * 1e3))
        print('%s: trace %d points, selected memsize %d' %
              (digitizer.name, npoints, memsize))
        print('%s: pre and post trigger: %d %d' % (digitizer.name,
                                                   digitizer.data_memory_size() - digitizer.posttrigger_memory_size(),
                                                   digitizer.posttrigger_memory_size()))
    return memsize


@qtt.pgeometry.static_var('debug_enabled', False)
@qtt.pgeometry.static_var('debug_data', {})
def measuresegment_m4i(digitizer, waveform, read_ch, mV_range, Naverage=100, process=False, verbose=0, fig=None, trigger_re_arm_compensation = False, trigger_re_arm_padding = True):
    """ Measure block data with M4i

    Args:
        digitizer (object): handle to instrument
        waveform (dict): waveform specification
        read_ch (list): channels to read from the instrument
        mV_range (float): range for input
        Naverage (int): number of averages to perform
        verbose (int): verbosity level
        trigger_re_arm_compensation (bool): Passed to raw measurement function
        trigger_re_arm_padding (bool):  Passed to raw measurement function
    Returns:
        data (numpy array): recorded and processed data

    """

    period = waveform['period']
    raw_data = measure_raw_segment_m4i(digitizer, period, read_ch, mV_range=mV_range,
                                       Naverage=Naverage, verbose=verbose, trigger_re_arm_compensation=trigger_re_arm_compensation, trigger_re_arm_padding = trigger_re_arm_padding)
    if measuresegment_m4i.debug_enabled:
        measuresegment_m4i.debug_data['raw_data'] = raw_data
        measuresegment_m4i.debug_data['waveform'] = waveform
        measuresegment_m4i.debug_data['timestamp'] = qtt.data.dateString()

    if not process:
        return raw_data

    samplerate = digitizer.sample_rate()
    if 'width' in waveform:
        width = [waveform['width']]
    else:
        width = [waveform['width_horz'], waveform['width_vert']]

    resolution = waveform.get('resolution', None)
    start_zero = waveform.get('start_zero', False)

    if len(width) == 2:
        data, _ = process_2d_sawtooth(raw_data.T, period, samplerate,
                                      resolution, width, start_zero=start_zero, fig=fig)
    else:

        data, _ = process_1d_sawtooth(raw_data.T, width, period, samplerate,
                                      resolution=resolution, start_zero=start_zero,
                                      verbose=verbose, fig=fig)
    if measuresegment_m4i.debug_enabled:
        measuresegment_m4i.debug_data['data'] = data
        
    if verbose:
        print('measuresegment_m4i: processed_data: width %s, data shape %s' % (width, data.shape,))

    return data


def get_uhfli_scope_records(device, daq, scopeModule, number_of_records=1, timeout=30):
    """
    Obtain scope records from the device using an instance of the Scope Module.
    """
    # Enable the scope: Now the scope is ready to record data upon receiving triggers.
    scopeModule.set('scopeModule/mode', 1)
    scopeModule.subscribe('/' + device + '/scopes/0/wave')
    daq.setInt('/%s/scopes/0/enable' % device, 1)
    scopeModule.execute()
    daq.sync()

    # Wait until the Scope Module has received and processed the desired number of records.
    start = time.time()
    records = 0
    progress = 0
    while (records < number_of_records) or (progress < 1.0):
        records = scopeModule.getInt("scopeModule/records")
        progress = scopeModule.progress()[0]
        if (time.time() - start) > timeout:
            # Break out of the loop if for some reason we're no longer receiving scope data from the device.
            logging.warning(
                "\nScope Module did not return {} records after {} s - forcing stop.".format(number_of_records, timeout))
            break
    daq.setInt('/%s/scopes/0/enable' % device, 0)
    # Read out the scope data from the module.
    data = scopeModule.read(True)
    # Stop the module; to use it again we need to call execute().
    scopeModule.finish()
    wave_nodepath = '/{}/scopes/0/wave'.format(device)
    return data[wave_nodepath][:number_of_records]


def measure_segment_uhfli(zi, waveform, channels, number_of_averages=100, **kwargs):
    """ Measure block data with Zurich Instruments UHFLI

    Args:
        zi (ZIUHFL): Instance of QCoDeS driver for  ZI UHF-LI
        waveform (dict): Information about the waveform that is to be collected
        channels (list): List of channels to read from, can be 1, 2 or both.
        number_of_averages (int) : Number of times the sample is collected
    Returns:
        data (numpy array): An array of arrays, one array per input channel.

    """
    period = waveform['period']
    zi.scope_duration.set(period)  # seconds

    if 1 in channels:
        zi.scope_channel1_input.set('Signal Input 1')
    if 2 in channels:
        zi.scope_channel2_input.set('Signal Input 2')
    zi.scope_channels.set(sum(channels))  # 1: Chan1 only, 2: Chan2 only, 3: Chan1 + Chan2

    if not zi.scope_correctly_built:
        zi.Scope.prepare_scope()

    scope_records = get_uhfli_scope_records(zi.device, zi.daq, zi.scope, 1)
    data = []
    for channel_index, _ in enumerate(channels):
        for _, record in enumerate(scope_records):
            wave = record[0]['wave'][channel_index, :]
            data.append(wave)
    avarage = np.average(np.asarray(data), axis=0)
    return [avarage]


def measure_segment_scope_reader(scope_reader, waveform, number_of_averages, process=True, **kwargs):
    """ Measure block data with scope reader.

    Args:
        scope_reader (AcquisitionScopeInterface): Instance of scope reader.
        waveform (dict): Information about the waveform that is to be collected.
        number_of_averages (int) : Number of times the sample is collected.
        process (bool): If True, cut off the downward sawtooth slopes from the data.

    Returns:
        data (numpy array): An array of arrays, one array per input channel.

    """
    data_arrays = scope_reader.acquire(number_of_averages)
    raw_data = np.array(data_arrays)
    if not process:
        return raw_data

    if 'width' in waveform:
        width = [waveform['width']]
    else:
        width = [waveform['width_horz'], waveform['width_vert']]

    resolution = waveform.get('resolution', None)
    start_zero = waveform.get('start_zero', False)
    sample_rate = scope_reader.sample_rate
    period = waveform['period']

    if len(width) == 2:
        data, _ = process_2d_sawtooth(raw_data.T, period, sample_rate, resolution,
                                      width, start_zero=start_zero, fig=None)
    else:
        data, _ = process_1d_sawtooth(raw_data.T, width, period, sample_rate,
                                      resolution=resolution, start_zero=start_zero, fig=None)

    return data


def measuresegment(waveform, Naverage, minstrhandle, read_ch, mV_range=2000, process=True, device_parameters = None):
    """Wrapper to identify measurement instrument and run appropriate acquisition function.
    Supported instruments: m4i digitizer, ZI UHF-LI

    Args:
        waveform (dict): waveform specification
        Naverage (int): number of averages to perform
        minstrhandle (str or Instrument): handle to acquisition device
        read_ch (list): channels to read from the instrument
        mV_range (float): range for input
        verbose (int): verbosity level
        device_parameters (dict): dictionary passed as keyword parameters to the measurement methods
    Returns:
        data (numpy array): recorded and processed data

    """
    if device_parameters is None:
        device_parameters = {}
        
    try:
        is_m4i = isinstance(minstrhandle, qcodes.instrument_drivers.Spectrum.M4i.M4i)
    except:
        is_m4i = False
    try:
        is_uhfli = isinstance(minstrhandle, qcodes.instrument_drivers.ZI.ZIUHFLI.ZIUHFLI)
    except:
        is_uhfli = False
    try:
        is_scope_reader = isinstance(minstrhandle, AcquisitionScopeInterface)
    except:
        is_scope_reader = False

    minstrument = get_instrument(minstrhandle)
    is_simulation = isinstance(minstrhandle, SimulationDigitizer)

    if is_m4i:
        data = measuresegment_m4i(minstrhandle, waveform, read_ch, mV_range, Naverage, process=process, **device_parameters)
    elif is_uhfli:
        data = measure_segment_uhfli(minstrhandle, waveform, read_ch, Naverage, **device_parameters)
    elif is_scope_reader:
        data = measure_segment_scope_reader(minstrhandle, waveform, Naverage, process=process, **device_parameters)
    elif minstrhandle == 'dummy':
        # for testing purposes
        data = np.random.rand(100, )
    elif is_simulation:
        data = minstrument.measuresegment(waveform, channels=read_ch)
    else:
        raise Exception('Unrecognized fast readout instrument %s' % minstrhandle)
    if np.array(data).size == 0:
        warnings.warn('measuresegment: received empty data array')
    return data


def acquire_segments(station, parameters, average=True, mV_range=2000,
                     save_to_disk=True, location=None, verbose=True, trigger_re_arm_compensation=False, trigger_re_arm_padding = True):
    """Record triggered segments as time traces into dataset. AWG must be already sending a trigger pulse per segment.

    Note that if the requested period is equal or longer than the period on the AWG, then not all trigger events might
    be used by the M4i.

    The saving to disk can take minutes or even longer.

    Args:
        parameters (dict): dictionary containing the following compulsory parameters:
            minstrhandle (instrument handle): measurement instrument handle (m4i digitizer).
            read_ch (list of int): channel numbers to record.
            period (float): time in seconds to record for each segment.
            nsegments (int): number of segments to record.
            average (bool): if True, dataset will contain a single time trace with the average of all acquired segments; 
                            if False, dataset will contain nsegments single time trace acquisitions.
            verbose (bool): print to the console.

    Returns:
        alldata (dataset): time trace(s) of the segments acquired.
    """
    minstrhandle = parameters['minstrhandle']
    read_ch = parameters['read_ch']
    period = parameters['period']
    nsegments = parameters['nsegments']

    t0 = time.time()

    waveform = {'period': period, 'width': 0}
    if isinstance(read_ch, int):
        read_ch = [read_ch]
    if len(read_ch) == 1:
        measure_names = ['measured']
    else:
        measure_names = ['READOUT_ch%d' % c for c in read_ch]

    if verbose:
        exepected_measurement_time = nsegments*period
        print(f'acquire_segments: expected measurement time: {exepected_measurement_time:.3f} [s]')

    ism4i = isinstance(
            minstrhandle, qcodes.instrument_drivers.Spectrum.M4i.M4i)
    if average:
        data = measuresegment(waveform, nsegments,
                              minstrhandle, read_ch, mV_range, process=False, device_parameters = {'trigger_re_arm_compensation': trigger_re_arm_compensation, 'trigger_re_arm_padding': trigger_re_arm_padding})
        if ism4i:
            segment_time = np.arange(0., len(data[0])) / minstrhandle.exact_sample_rate()
        else:
            segment_time = np.linspace(0, period, len(data[0]))
        alldata = makeDataSet1Dplain('time', segment_time, measure_names, data,
                                     xunit='s', location=location, loc_record={'label': 'acquire_segments'})
    else:
        if ism4i:
            memsize_total, pre_trigger, signal_start, signal_end = select_m4i_memsize(
                minstrhandle, period, trigger_delay=None, nsegments=nsegments, verbose=verbose >= 2, trigger_re_arm_compensation = trigger_re_arm_compensation)

            segment_size = int(memsize_total / nsegments)
            post_trigger = segment_size - pre_trigger

            if mV_range is None:
                mV_range = minstrhandle.range_channel_0()

            minstrhandle.initialize_channels(read_ch, mV_range=mV_range,
                                             memsize=minstrhandle._channel_memsize, termination=None)

            sample_rate = minstrhandle.exact_sample_rate()
            dataraw = minstrhandle.multiple_trigger_acquisition(
                mV_range, memsize_total, seg_size=segment_size, posttrigger_size=post_trigger)
            if np.all(dataraw == 0):
                warnings.warn('multiple_trigger_acquisition returned zero data! did a timeout occur?')
            if isinstance(dataraw, tuple):
                dataraw = dataraw[0]
            data = np.reshape(np.transpose(np.reshape(dataraw, (-1, len(read_ch)))), (len(read_ch), nsegments, -1))
            data = data[:, :, signal_start:signal_end]
            if trigger_re_arm_compensation and trigger_re_arm_padding:
                data = _trigger_re_arm_padding(data, signal_end-signal_start)
            segment_time = np.arange(0., data.shape[2]) / sample_rate
            segment_num = np.arange(nsegments).astype(segment_time.dtype)
            alldata = makeDataSet2Dplain('time', segment_time, 'segment_number', segment_num,
                                         zname=measure_names, z=data, xunit='s', location=location,
                                         loc_record={'label': 'acquire_segments'})
        else:
            raise Exception(f'Non-averaged acquisitions not supported for measurement instrument {minstrhandle}')

    dt = time.time() - t0
    update_dictionary(alldata.metadata, dt=dt, station=station.snapshot())
    update_dictionary(alldata.metadata, scantime=str(
        datetime.datetime.now()), nsegments=str(nsegments))

    if verbose:
        print(f'acquire_segments: acquired data of shape {data.shape}')
    if hasattr(station, 'gates'):
        gates = station.gates
        gatevals = gates.allvalues()
        update_dictionary(alldata.metadata, allgatevalues=gatevals)

    if save_to_disk:
        alldata = qtt.utilities.tools.stripDataset(alldata)
        alldata.write(write_metadata=True)

    return alldata


def single_shot_readout(minstparams, length, shots, threshold=None):
    """Acquires several measurement traces, averages the signal over the entire trace for each shot and returns the proportion of shots that are above a defined threshold.
    NOTE: The AWG marker delay should be set so that the triggered acquisition starts at the correct part of the readout pulse.

    Args:
        minstparams (dict): required parameters of the digitizer (handle, read_ch, mV_range)
        length (float): length of each shot, in seconds
        shots (int): number of shots to acquire
        threshold (float): signal discrimination threshold. If None, readout proportion is not calculated.

    Returns:
        proportion (float [0,1]): proportion of shots above the threshold
        allshots (array of floats): average signal of every shot taken
    """
    minstrhandle = minstparams['handle']
    if not isinstance(minstrhandle, qcodes.instrument_drivers.Spectrum.M4i.M4i):
        raise (Exception('single shot readout is only supported for M4i digitizer'))
    read_ch = minstparams['read_ch']
    if isinstance(read_ch, int):
        read_ch = [read_ch]
    if len(read_ch) > 1:
        raise (Exception('cannot do single shot readout with multiple channels'))
    mV_range = minstparams.setdefault('mV_range', 2000)
    memsize = select_digitizer_memsize(minstrhandle, length, nsegments=shots, verbose=0)
    post_trigger = minstrhandle.posttrigger_memory_size()
    minstrhandle.initialize_channels(read_ch, mV_range=mV_range, memsize=memsize)
    dataraw = minstrhandle.multiple_trigger_acquisition(mV_range, memsize, memsize // shots, post_trigger)
    data = np.reshape(dataraw, (shots, -1))
    allshots = np.mean(data, 1)
    if threshold is None:
        proportion = None
    else:
        proportion = sum(allshots > threshold) / shots

    return proportion, allshots


# %%

def scan2Dfast(station, scanjob, location=None, liveplotwindow=None, plotparam='measured',
               diff_dir=None, verbose=1, extra_metadata=None):
    """Make a 2D scan and create qcodes dataset to store on disk.

    Args:
        station (object): contains all the instruments
        scanjob (scanjob_t): data for scan
        extra_metadata (None or dict): additional metadata to be included in the dataset

    Returns:
        alldata (DataSet): contains the measurement data and metadata
    """
    gates = station.gates
    gatevals = gates.allvalues()

    scanjob.check_format()

    if 'sd' in scanjob:
        warnings.warn('sd argument is not supported in scan2Dfast')

    if type(scanjob) is dict:
        warnings.warn('Use the scanjob_t class.', DeprecationWarning)
        scanjob = scanjob_t(scanjob)

    scanjob._parse_stepdata('stepdata', gates=gates)
    scanjob._parse_stepdata('sweepdata', gates=gates)

    scanjob.parse_param('sweepdata', station, paramtype='fast')
    scanjob.parse_param('stepdata', station, paramtype='slow')

    minstrhandle = qtt.measurements.scans.get_instrument(scanjob.get('minstrumenthandle', 'digitizer'))

    read_ch = get_minstrument_channels(scanjob['minstrument'])
    virtual_awg = getattr(station, 'virtual_awg', None)

    if isinstance(scanjob['stepdata']['param'], lin_comb_type) or isinstance(scanjob['sweepdata']['param'],
                                                                             lin_comb_type):
        scanjob['scantype'] = 'scan2Dfastvec'
        fast_sweep_gates = scanjob['sweepdata']['param'].copy()
        if 'stepvalues' in scanjob:
            scanjob._start_end_to_range(scanfields=['sweepdata'])
        else:
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
        if 'pulsedata' in scanjob:
            sg = []
            for g, v in fast_sweep_gates.items():
                if v != 0:
                    sg.append(g)
            if len(sg) > 1:
                raise(Exception('AWG pulses does not yet support virtual gates'))
            waveform, _ = station.awg.sweepandpulse_gate({'gate': sg[0], 'sweeprange': sweepdata['range'],
                                                          'period': period}, scanjob['pulsedata'])
        else:
            if virtual_awg:
                sweep_range = sweepdata['range']
                waveform = virtual_awg.sweep_gates(fast_sweep_gates, sweep_range, period)
                virtual_awg.enable_outputs(list(fast_sweep_gates.keys()))
                virtual_awg.run()
            else:
                waveform, sweep_info = station.awg.sweep_gate_virt(fast_sweep_gates, sweepdata['range'], period)
    else:
        if 'range' in sweepdata:
            sweeprange = sweepdata['range']
        else:
            sweeprange = (sweepdata['end'] - sweepdata['start'])
            sweepgate_value = (sweepdata['start'] + sweepdata['end']) / 2
            gates.set(sweepdata['paramname'], float(sweepgate_value))
        if 'pulsedata' in scanjob:
            waveform, sweep_info = station.awg.sweepandpulse_gate({'gate': sweepdata['param'].name, 'sweeprange': sweeprange,
                                                                   'period': period}, scanjob['pulsedata'])
        else:
            if virtual_awg:
                gates = {sweepdata['param'].name: 1}
                waveform = virtual_awg.sweep_gates(gates, sweeprange, period)
                virtual_awg.enable_outputs(list(gates.keys()))
                virtual_awg.run()
            else:
                waveform, sweep_info = station.awg.sweep_gate(sweepdata['param'].name, sweeprange, period)

    data = measuresegment(waveform, Naverage, minstrhandle, read_ch)
    if len(read_ch) == 1:
        measure_names = ['measured']
    else:
        measure_names = ['READOUT_ch%d' % c for c in read_ch]
        if plotparam == 'measured':
            plotparam = measure_names[0]

    scanvalues = scanjob._convert_scanjob_vec(station, data[0].shape[0], stepvalues=scanjob.get('stepvalues', None))
    stepvalues = scanvalues[0]
    sweepvalues = scanvalues[1]

    logging.info('scan2D: %d %d' % (len(stepvalues), len(sweepvalues)))
    logging.info('scan2D: wait_time %f' % wait_time)
    t0 = time.time()

    if type(stepvalues) is np.ndarray:
        if stepvalues.ndim > 1:
            stepvalues_tmp = stepdata['param'].params[0][list(stepvalues[:, 0])]
        else:
            stepvalues_tmp = stepdata['param'][list(stepvalues[:, 0])]
        # added to overwrite array names for setpoint arrays
        if 'paramname' in sweepdata:
            stepvalues_tmp.name = sweepdata['paramname']
        if 'paramname' in stepdata:
            stepvalues_tmp.name = stepdata['paramname']
        alldata = makeDataSet2D(stepvalues_tmp, sweepvalues, measure_names=measure_names,
                                location=location, loc_record={'label': scanjob['scantype']})
    else:
        if stepvalues.name == sweepvalues.name:
            stepvalues.name = stepvalues.name + '_y'
            sweepvalues.name = sweepvalues.name + '_x'
        alldata = makeDataSet2D(stepvalues, sweepvalues, measure_names=measure_names,
                                location=location, loc_record={'label': scanjob['scantype']})

    liveplotwindow = _initialize_live_plotting(alldata, plotparam, liveplotwindow, subplots=True)

    tprev = time.time()

    for ix, x in enumerate(stepvalues):
        if type(stepvalues) is np.ndarray:
            tprint('scan2Dfast: %d/%d: setting %s to %s' %
                   (ix, len(stepvalues), stepdata['param'].name, str(x)), dt=.5)
        else:
            tprint('scan2Dfast: %d/%d: setting %s to %.3f' %
                   (ix, len(stepvalues), stepvalues.name, x), dt=.5)
        if scanjob['scantype'] == 'scan2Dfastvec' and isinstance(stepdata['param'], dict):
            for g in stepdata['param']:
                gates.set(g, (scanjob['phys_gates_vals'][g][ix, 0]
                              + scanjob['phys_gates_vals'][g][ix, -1]) / 2)
        else:
            stepdata['param'].set(x)
        if ix == 0:
            time.sleep(wait_time_startscan)
        else:
            time.sleep(wait_time)
        data = measuresegment(waveform, Naverage, minstrhandle, read_ch)
        for idm, mname in enumerate(measure_names):
            alldata.arrays[mname].ndarray[ix] = data[idm]

        delta, tprev, update = _delta_time(tprev, thr=1.)
        if update:
            if liveplotwindow is not None:
                liveplotwindow.update_plot()
            pyqtgraph.mkQApp().processEvents()
        if qtt.abort_measurements():
            print('  aborting measurement loop')
            break

    if virtual_awg:
        virtual_awg.stop()
    else:
        station.awg.stop()

    dt = time.time() - t0

    if liveplotwindow is not None:
        liveplotwindow.update_plot()
        pyqtgraph.mkQApp().processEvents()

    if hasattr(stepvalues, 'ndim') and stepvalues.ndim > 1:
        for idp, steppm_add in enumerate(stepdata['param'].params):
            if idp <= 0:
                continue
            data_arr_step_add = DataArray(steppm_add, name=steppm_add.name, full_name=steppm_add.name, array_id=steppm_add.name,
                                          preset_data=np.repeat(
                                              stepvalues[:, idp, np.newaxis], alldata.arrays[measure_names[0]].shape[1], axis=1),
                                          set_arrays=(alldata.arrays[measure_names[0]].set_arrays))
            alldata.add_array(data_arr_step_add)

    if diff_dir is not None:
        for mname in measure_names:
            alldata = diffDataset(alldata, diff_dir=diff_dir,
                                  fig=None, meas_arr_name=mname)

    if not hasattr(alldata, 'metadata'):
        alldata.metadata = dict()

    if extra_metadata is not None:
        update_dictionary(alldata.metadata, **extra_metadata)

    update_dictionary(alldata.metadata, scanjob=dict(scanjob), allgatevalues=gatevals,
                      dt=dt, station=station.snapshot())
    _add_dataset_metadata(alldata)

    alldata.write(write_metadata=True)

    return alldata


def create_vectorscan(virtual_parameter, g_range=1, sweeporstepdata=None, remove_slow_gates=False, station=None,
                      start=0, step=None):
    """Converts the sweepdata or stepdata of a scanjob in those needed for virtual vector scans

    Args:
        virtual_parameter (obj): parameter of the virtual gate which is varied
        g_range (float): scan range (total range)
        remove_slow_gates: Removes slow gates from the linear combination of gates. Useful if virtual gates include compensation ofn slow gates, but a fast measurement should be run.
        start (float): start if the scanjob data 
        step (None or float): if not None, then add to the scanning field
    Returns:
        sweeporstepdata (dict): sweepdata or stepdata needed in the scanjob for virtual vector scans
    """
    if sweeporstepdata is None:
        sweeporstepdata = {}
    if hasattr(virtual_parameter, 'comb_map'):
        pp = dict([(p.name, r)
                   for p, r in virtual_parameter.comb_map if round(r, 5) != 0])
        if remove_slow_gates:
            try:
                for gate in list(pp.keys()):
                    if not station.awg.awg_gate(gate):
                        pp.pop(gate, None)

            except BaseException:
                warnings.warn(f'error when removing slow gate {gate} from scan data')
    else:
        pp = {virtual_parameter.name: 1}
    sweeporstepdata = {'start': start, 'range': g_range,
                       'end': start + g_range, 'param': pp}
    if step is not None:
        sweeporstepdata['step'] = step
    return sweeporstepdata


def plotData(alldata, diff_dir=None, fig=1):
    """ Plot a dataset and optionally differentiate """
    figure = plt.figure(fig)
    plt.clf()
    if diff_dir is not None:
        imx = qtt.utilities.tools.diffImageSmooth(alldata.measured.ndarray, dy=diff_dir)
        name = 'diff_dir_%s' % diff_dir
        name = uniqueArrayName(alldata, name)
        data_arr = qcodes.DataArray(name=name, label=name, array_id=name,
                                    set_arrays=alldata.measured.set_arrays, preset_data=imx)
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
            plot.fig.axes[1].autoscale(tight=True)
        except Exception as ex:
            logging.debug('autoscaling failed')


# %%


def scan2Dturbo(station, scanjob, location=None, liveplotwindow=None, delete=True, verbose=1):
    """Perform a very fast 2d scan by varying two physical gates with the AWG.

    The function assumes the station contains an acquisition device that is supported by the measuresegment function.
    The number of the measurement channels is supplied via the minstrument field in the scanjob.

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

    scanjob._parse_stepdata('stepdata', gates=gates)
    scanjob._parse_stepdata('sweepdata', gates=gates)

    scanjob.parse_param('sweepdata', station, paramtype='fast')
    scanjob.parse_param('stepdata', station, paramtype='fast')

    minstrhandle = qtt.measurements.scans.get_instrument(scanjob.get('minstrumenthandle', 'digitizer'))
    virtual_awg = getattr(station, 'virtual_awg', None)
    read_ch = get_minstrument_channels(scanjob['minstrument'])
    sweep_info = None

    if isinstance(read_ch, int):
        read_ch = [read_ch]

    if isinstance(scanjob['stepdata']['param'], lin_comb_type) or isinstance(scanjob['sweepdata']['param'],
                                                                             lin_comb_type):
        scanjob['scantype'] = 'scan2Dturbovec'
        fast_sweep_gates = scanjob['sweepdata']['param'].copy()
        fast_step_gates = scanjob['stepdata']['param'].copy()
        scanjob._start_end_to_range()
    else:
        scanjob['scantype'] = 'scan2Dturbo'

    stepdata = scanjob['stepdata']
    sweepdata = scanjob['sweepdata']

    Naverage = scanjob.get('Naverage', 20)
    resolution = scanjob.get('resolution', [96, 96])

    t0 = time.time()

    wait_time_startscan = scanjob.get('wait_time_startscan', 0)

    if scanjob['scantype'] == 'scan2Dturbo' and 'start' in sweepdata:
        stepdata['param'].set((stepdata['end'] + stepdata['start']) / 2)
        sweepdata['param'].set((sweepdata['end'] + sweepdata['start']) / 2)
        sweepranges = [sweepdata['end'] - sweepdata['start'],
                       stepdata['end'] - stepdata['start']]
    else:
        sweepranges = [sweepdata['range'], stepdata['range']]

    try:
        ism4i = isinstance(
            minstrhandle, qcodes.instrument_drivers.Spectrum.M4i.M4i)
    except:
        ism4i = False
    if ism4i:
        samp_freq = minstrhandle.sample_rate()
        resolution[0] = np.ceil(resolution[0] / 16) * 16
    else:
        raise Exception(
            'Unrecognized fast readout instrument %s' % minstrhandle)

    if scanjob['scantype'] == 'scan2Dturbo':
        sweepgates = [sweepdata['param'].name, stepdata['param'].name]
        if virtual_awg:
            period = np.prod(resolution) / samp_freq
            sweep_gates = [{g: 1} for g in sweepgates]
            waveform = virtual_awg.sweep_gates_2d(sweep_gates, sweepranges, period, resolution, do_upload=delete)
            virtual_awg.enable_outputs(sweepgates)
            virtual_awg.run()
        else:
            waveform, sweep_info = station.awg.sweep_2D(samp_freq, sweepgates, sweepranges, resolution, delete=delete)
        if verbose:
            print('scan2Dturbo: sweepgates %s' % (str(sweepgates),))
    else:
        scanjob._parse_2Dvec()
        if virtual_awg:
            sweepgates = [*fast_sweep_gates, *fast_step_gates]
            period = np.prod(resolution) / samp_freq
            sweep_gates = [{g: 1 for g in fast_sweep_gates}, {g: 1 for g in fast_step_gates}]
            waveform = virtual_awg.sweep_gates_2d(sweep_gates, sweepranges, period, resolution, do_upload=delete)
            virtual_awg.enable_outputs(sweepgates)
            virtual_awg.run()
        else:
            waveform, sweep_info = station.awg.sweep_2D_virt(samp_freq, fast_sweep_gates, fast_step_gates, sweepranges,
                                                             resolution, delete=delete)

    time.sleep(wait_time_startscan)
    data = measuresegment(waveform, Naverage, minstrhandle, read_ch)
    scan2Dturbo._data = data

    if virtual_awg:
        virtual_awg.disable_outputs(sweepgates)
        virtual_awg.stop()
    else:
        station.awg.stop()

    if len(read_ch) == 1:
        measure_names = ['measured']
    else:
        measure_names = ['READOUT_ch%d' % c for c in read_ch]

    if scanjob['scantype'] == 'scan2Dturbo':
        alldata, _ = makeDataset_sweep_2D(data, gates, sweepgates, sweepranges, measure_names=measure_names,
                                          location=location, loc_record={'label': scanjob['scantype']})
    else:
        scanvalues = scanjob._convert_scanjob_vec(
            station, data[0].shape[1], data[0].shape[0])
        stepvalues = scanvalues[0]
        sweepvalues = scanvalues[1]
        alldata = makeDataSet2D(stepvalues, sweepvalues, measure_names=measure_names,
                                preset_data=data, location=location, loc_record={'label': scanjob['scantype']})

    dt = time.time() - t0
    liveplotwindow = _initialize_live_plotting(alldata, plotparam=None, liveplotwindow=liveplotwindow)

    if not hasattr(alldata, 'metadata'):
        alldata.metadata = dict()

    update_dictionary(alldata.metadata, scanjob=dict(scanjob),
                      dt=dt, station=station.snapshot(), allgatevalues=gatevals)
    _add_dataset_metadata(alldata)
    alldata.write(write_metadata=True)
    return alldata, waveform, sweep_info


# %% Measurement tools


def waitTime(gate, station=None, gate_settle=None, default=1e-3):
    """ Return settle times for gates on a station """
    if gate is None:
        return 0.001
    if gate_settle is not None:
        return gate_settle(gate)
    if station is not None:
        if hasattr(station, 'gate_settle'):
            return station.gate_settle(gate)
    return default


# %%


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
            sweepgate_param = getattr(gates, sweepgate)
            sweepgate_value = sweepgate_param.get()
        else:
            raise Exception('No gates supplied')

    if isinstance(ynames, list):
        sweeplength = len(data[0])
    else:
        sweeplength = len(data)
    sweepvalues = np.linspace(
        sweepgate_value - sweeprange / 2, sweepgate_value + sweeprange / 2, sweeplength)

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


def makeDataset_sweep_2D(data, gates, sweepgates, sweepranges, measure_names='measured', location=None, loc_record=None,
                         fig=None):
    """Convert the data of a 2D sweep to a DataSet."""

    scantype = loc_record['label']

    vector_scan = False
    if 'vec' in scantype:
        vector_scan = True
    if isinstance(sweepgates, dict):
        vector_scan = True
    if isinstance(sweepgates, list) and isinstance(sweepgates[0], dict):
        vector_scan = True

    if not vector_scan:
        # simple gate type
        gate_horz = getattr(gates, sweepgates[0])
        gate_vert = getattr(gates, sweepgates[1])

        initval_horz = gate_horz.get()
        initval_vert = gate_vert.get()

        if type(measure_names) is list:
            data_measured = data[0]
        else:
            data_measured = data

        sweep_horz = gate_horz[initval_horz - sweepranges[0]
                               / 2:sweepranges[0] / 2 + initval_horz:sweepranges[0] / len(data_measured[0])]
        sweep_vert = gate_vert[initval_vert - sweepranges[1]
                               / 2:sweepranges[1] / 2 + initval_vert:sweepranges[1] / len(data_measured)]
    else:
        # vector scan
        gate_horz = 'gate_horz'
        gate_vert = 'gate_vert'
        p1 = qcodes.Parameter('gate_horz', set_cmd=None)
        p2 = qcodes.Parameter('gate_vert', set_cmd=None)

        sweepranges[0]
        xvals = np.linspace(-sweepranges[0] / 2, sweepranges[0] / 2, data.shape[1])
        yvals = np.linspace(-sweepranges[1] / 2, sweepranges[1] / 2, data.shape[0])

        sweep_horz = p1[xvals]
        sweep_vert = p2[yvals]

        assert (data.shape[0] == len(list(sweep_vert)))
        assert (data.shape[1] == len(list(sweep_horz)))

    dataset = makeDataSet2D(sweep_vert, sweep_horz, measure_names=measure_names,
                            location=location, loc_record=loc_record, preset_data=data)

    if fig is None:
        return dataset, None
    else:
        if fig is not None:
            plt.figure(fig).clear()
        plot = MatPlot(dataset.measured, interval=0, num=fig)
        return dataset, plot


# %%


@qtt.utilities.tools.rdeprecated(txt='Method will be removed in future release of qtt.', expire='1 Sep 2018')
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
                  (g, adata['pinchoff_point']))
        pv[ii] = adata['pinchoff_point']
    od['pinchvalues'] = pv
    return od


def enforce_boundaries(scanjob, sample_data, eps=1e-2):
    """ Make sure a scanjob does not go outside sample boundaries

    Args:
        scanjob (scanjob_t or dict)
        sample_data (sample_data_t)
    """
    if isinstance(scanjob, scanjob_t) or ('minstrument' in scanjob):
        for field in ['stepdata', 'sweepdata']:

            if field in scanjob:
                bstep = sample_data.gate_boundaries(scanjob[field]['param'])
                scanjob[field]['end'] = max(
                    scanjob[field]['end'], bstep[0] + eps)
                scanjob[field]['start'] = max(
                    scanjob[field]['start'], bstep[0] + eps)
                scanjob[field]['end'] = min(
                    scanjob[field]['end'], bstep[1] - eps)
                scanjob[field]['start'] = min(
                    scanjob[field]['start'], bstep[1] - eps)
    else:
        for param in scanjob:
            bstep = sample_data.gate_boundaries(param)
            scanjob[param] = max(scanjob[param], bstep[0] + eps)
            scanjob[param] = min(scanjob[param], bstep[1] - eps)
