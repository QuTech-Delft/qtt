"""
Contains code for various structures

"""
import numpy as np
import matplotlib.pyplot as plt
import copy
from functools import partial

import qcodes
import qcodes.instrument.parameter

import qtt
import qtt.measurements.scans
from qtt.algorithms.coulomb import peakdataOrientation, coulombPeaks, findSensingDotPosition
from qtt.utilities.tools import freezeclass
from qtt.dataset_processing import process_dataarray


# %%


@freezeclass
class twodot_t(dict):

    def __init(self, gates, name=None):
        """ Class to represent a double quantum dot

        Args:
            gates (list of str): gate names of barrier left, plunger left, barrier middle, plunger right and barrier
                                 right
            name (str): name for the object
        """
        self['gates'] = gates
        self['name'] = name

    def name(self):
        return self['name']

    def __repr__(self):
        s = '%s: %s at 0x%x' % (self.__class__.__name__, self.name(), id(self))
        return s

    def __getstate__(self):
        """ Helper function to allow pickling of object """
        d = {}
        for k, v in self.__dict__.items():
            if k not in ['station']:
                d[k] = copy.deepcopy(v)
        return d


@freezeclass
class onedot_t(dict):
    """ Class representing a single quantum dot """

    def __init__(self, gates, name=None, data=None, station=None, transport_instrument=None):
        """ Class to represent a single quantum dot

        Args:
            gates (list of str): names of gates to use for left barrier, plunger and right barrier
            name (str): name for the object
            transport_instrument (str or Instrument): instrument to use for transport measurements
            data (dict or None): data for internal storage
            station (obj): object with references to instruments

        """
        self['gates'] = gates
        self.station = station
        self['transport_instrument'] = transport_instrument
        self['instrument'] = transport_instrument  # legacy code
        if name is None:
            name = 'dot-%s' % ('-'.join(gates))
        self['name'] = name

        if data is None:
            data = {}
        self.data = data

    def name(self):
        return self['name']

    def __repr__(self):
        s = '%s: %s at 0x%x' % (self.__class__.__name__, self.name(), id(self))
        return s

    def __getstate__(self):
        """ Helper function to allow pickling of object """
        d = {}
        import copy
        for k, v in self.__dict__.items():
            if k not in ['station']:
                d[k] = copy.deepcopy(v)
        return d


# %%


def _scanlabel(ds):
    """ Helper function """
    a = ds.default_parameter_array()
    lx = a.set_arrays[0].label
    unit = a.set_arrays[0].unit
    if unit:
        lx += '[' + unit + ']'
    return lx


@freezeclass
class sensingdot_t:

    def __init__(self, gate_names, gate_values=None, station=None, index=None, minstrument=None, virt_gates=None):
        """ Class representing a sensing dot

        We assume the sensing dot can be controlled by two barrier gates and a single plunger gate.
        An instrument to measure the current through the dot is provided by the minstrument argument.

        Args:
            gate_names (list): gates to be used
            gate_values (array or None): values to be set on the gates
            station (Qcodes station)
            minstrument (tuple or str or Parameter): measurement instrument to use. tuple of instrument and channel
                                                     index
            index (None or int): deprecated
            fpga_ch (deprecated, int): index of FPGA channel to use for readout
            virt_gates (None or object): virtual gates object (optional)
            boxcar_filter_kernel_size (int): size of boxcar filter kernel to use in post-processing
        """
        self.verbose = 1
        self.gg = gate_names
        if gate_values is None:
            gate_values = [station.gates.get(g) for g in self.gg]
        self.sdval = gate_values
        self.targetvalue = np.NaN

        self.boxcar_filter_kernel_size = 1

        self._selected_peak = None
        self._detune_axis = np.array([1, -1])
        self._detune_axis = self._detune_axis / np.linalg.norm(self._detune_axis)
        self._debug = {}  # store debug data
        self.name = '-'.join([s for s in self.gg])
        self.goodpeaks = None
        self.station = station
        self.index = index
        self.minstrument = minstrument
        if index is not None:
            raise Exception('use minstrument argument')
        self.virt_gates = virt_gates

        self.data = {}

        self.plunger = qcodes.Parameter('plunger', get_cmd=self.plungervalue, set_cmd=self._set_plunger)

        # values for measurement
        if index is not None:
            self.valuefunc = station.components[
                'keithley%d' % index].amplitude.get
        else:
            self.valuefunc = None

    def __repr__(self):
        return 'sd gates: %s, %s, %s' % (self.gg[0], self.gg[1], self.gg[2])

    def __getstate__(self):
        """ Override to make the object pickable."""
        if self.verbose:
            print('sensingdot_t: __getstate__')
        import copy
        d = copy.copy(self.__dict__)
        for name in ['station', 'valuefunc']:
            if name in d:
                d[name] = str(d[name])
        return d

    def gates(self):
        """ Return values on the gates used to the define the SD """
        return self.sdval

    def gate_names(self):
        """ Return names of the gates used to the define the SD """
        return self.gg

    def show(self):
        gates = self.station.gates
        s = 'sensingdot_t: %s: %s: g %.1f, value %.1f/%.1f' % (
            self.gg[1], str(self.sdval), gates.get(self.gg[1]), self.value(), self.targetvalue)
        return s

    def initialize(self, sdval=None, setPlunger=False):
        gates = self.station.gates
        if sdval is not None:
            self.sdval = sdval
        gg = self.gg
        for ii in [0, 2]:
            gates.set(gg[ii], self.sdval[ii])
        if setPlunger:
            ii = 1
            gates.set(gg[ii], self.sdval[ii])

    def tunegate(self):
        """Return the gate used for tuning the potential in the dot """
        return self.gg[1]

    def plungervalue(self):
        """ Return current value of the chemical potential plunger """
        gates = self.station.gates
        return gates.get(self.tunegate())

    def _set_plunger(self, value):
        gates = self.station.gates
        gates.set(self.tunegate(), value)

    def value(self):
        """Return current through sensing dot """
        if self.valuefunc is not None:
            return self.valuefunc()
        raise Exception(
            'value function is not defined for this sensing dot object')

    def scan1D(sd, outputdir=None, step=-2., max_wait_time=.75, scanrange=300):
        """ Make 1D-scan of the sensing dot."""
        if sd.verbose:
            print('### sensing dot scan')
        minstrument = sd.minstrument
        if sd.index is not None:
            minstrument = [sd.index]
        gg = sd.gg
        sdval = sd.sdval
        gates = sd.station.gates

        for ii in [0, 2]:
            gates.set(gg[ii], sdval[ii])

        startval = sdval[1] + scanrange
        endval = sdval[1] - scanrange
        wait_time = 0.8
        try:
            wait_time = sd.station.gate_settle(gg[1])
        except BaseException:
            pass
        wait_time = np.minimum(wait_time, max_wait_time)

        scanjob1 = qtt.measurements.scans.scanjob_t()
        scanjob1['sweepdata'] = dict(
            {'param': gg[1], 'start': startval, 'end': endval, 'step': step, 'wait_time': wait_time})
        scanjob1['wait_time_startscan'] = .2 + 3 * wait_time
        scanjob1['minstrument'] = minstrument
        scanjob1['compensateGates'] = []
        scanjob1['gate_values_corners'] = [[]]

        if sd.verbose:
            print('sensingdot_t: scan1D: gate %s, wait_time %.3f' % (sd.gg[1], wait_time))
        alldata = qtt.measurements.scans.scan1D(sd.station, scanjob=scanjob1, verbose=sd.verbose)
        return alldata

    def detuning_scan(sd, stepsize=2, nsteps=5, verbose=1, fig=None):
        """ Optimize the sensing dot by making multiple plunger scans for different detunings

        Args:
            stepsize (float)
            nsteps (int)
        Returns:
            best (list): list of optimal detuning and sd plunger value
            results (dict)
        """

        gates = sd.station.gates
        gv0 = gates.allvalues()
        detunings = stepsize * np.arange(-(nsteps - 1) / 2, nsteps / 2)
        dd = []
        pp = []
        for ii, dt in enumerate(detunings):
            if verbose:
                print('detuning_scan: iteration %d: detuning %.3f' % (ii, dt))
            gates.resetgates(sd.gate_names(), gv0, verbose=0)
            sd.detune(dt)
            p, result = sd.fastTune(fig=None, verbose=verbose >= 2)
            dd.append(result)
            pp.append(sd.goodpeaks[0])
        gates.resetgates(sd.gate_names(), gv0, verbose=0)

        scores = [p['score'] for p in pp]
        bestidx = np.argmax(scores)
        optimal = [detunings[bestidx], pp[bestidx]['x']]
        if verbose:
            print('detuning_scan: best %d: detuning %.3f' % (bestidx, optimal[0]))
        results = {'detunings': detunings, 'scores': scores, 'bestpeak': pp[bestidx]}

        if fig:
            plt.figure(fig)
            plt.clf()
            for ii in range(len(detunings)):
                ds = dd[ii]
                p = pp[ii]
                y = ds.default_parameter_array()
                x = y.set_arrays[0]
                plt.plot(x, y, '-', label='scan %d: score %.3f' % (ii, p['score']))
                xx = np.array([p['x'], p['y']]).reshape((2, 1))
                qtt.pgeometry.plotLabels(xx, [scores[ii]])
                plt.title('Detuning scans for %s' % sd.__repr__())
                plt.xlabel(_scanlabel(result))
            plt.legend(numpoints=1)
            if verbose >= 2:
                plt.figure(fig + 1)
                plt.clf()
                plt.plot(detunings, scores, '.', label='Peak scores')
                plt.xlabel('Detuning [mV?]')
                plt.ylabel('Score')
                plt.title('Best peak scores for different detunings')

        return optimal, results

    def detune(self, value):
        """ Detune the sensing dot by the specified amount """

        gl = getattr(self.station.gates, self.gg[0])
        gr = getattr(self.station.gates, self.gg[2])
        gl.increment(self._detune_axis[0] * value)
        gr.increment(self._detune_axis[1] * value)

    def scan2D(sd, ds=90, stepsize=4, fig=None, verbose=1):
        """Make 2D-scan of the sensing dot."""
        gv = sd.station.gates.allvalues()

        gg = sd.gg
        sdval = sd.sdval

        sd.station.gates.set(gg[1], sdval[1])

        scanjob = qtt.measurements.scans.scanjob_t()
        scanjob['stepdata'] = dict(
            {'param': gg[0], 'start': sdval[0] + ds, 'end': sdval[0] - ds, 'step': stepsize})
        scanjob['sweepdata'] = dict(
            {'param': gg[2], 'start': sdval[2] + ds, 'end': sdval[2] - ds, 'step': stepsize})
        scanjob['minstrument'] = sd.minstrument

        if sd.verbose >= 1:
            print('sensing dot %s: performing barrier-barrier scan' % (sd,))
            if verbose >= 2:
                print(scanjob)
        alldata = qtt.measurements.scans.scan2D(sd.station, scanjob, verbose=verbose >= 2)

        sd.station.gates.resetgates(gv, gv, verbose=0)

        if fig is not None:
            qtt.measurements.scans.plotData(alldata, fig=fig)
        return alldata

    def _select_results(self, goodpeaks):
        if len(goodpeaks) > 0:
            self.sdval[1] = float(goodpeaks[0]['xhalfl'])
            self.targetvalue = float(goodpeaks[0]['yhalfl'])
            self._selected_peak = goodpeaks[0]
        else:
            self._selected_peak = None
            raise qtt.exceptions.CalibrationException('could not find good peak')

    def autoTune(sd, scanjob=None, fig=200, outputdir=None, step=-2.,
                 max_wait_time=1., scanrange=300, add_slopes=False):
        """ Automatically determine optimal value of plunger """
        if scanjob is not None:
            sd.autoTuneInit(scanjob)

        if sd.virt_gates is not None:
            raise Exception('virtual gates for slow scan not supported')

        alldata = sd.scan1D(outputdir=outputdir, step=step,
                            scanrange=scanrange, max_wait_time=max_wait_time)

        alldata = sd._measurement_post_processing(alldata)
        goodpeaks = sd._process_scan(alldata, useslopes=add_slopes, fig=fig)

        sd._select_results(goodpeaks)

        if sd.verbose:
            print('sensingdot_t: autotune complete: value %.1f [mV]' % sd.sdval[1])
        return sd.sdval[1], alldata

    def _process_scan(self, alldata, useslopes=True, fig=None, invert=False, verbose=0):
        """ Determine peaks in 1D scan """
        scan_sampling_rate = float(np.abs(alldata.metadata['scanjob']['sweepdata']['step']))
        x, y = qtt.data.dataset1Ddata(alldata)
        x, y = peakdataOrientation(x, y)

        if invert:
            y = -y

        if useslopes:
            goodpeaks = findSensingDotPosition(
                x, y, useslopes=useslopes, fig=fig, verbose=verbose, sampling_rate=scan_sampling_rate)
        else:

            goodpeaks = coulombPeaks(
                x, y, verbose=verbose, fig=fig, plothalf=True, sampling_rate=scan_sampling_rate)
        if fig is not None:
            plt.xlabel('%s' % (self.tunegate(),))
            plt.ylabel('%s' % (self.minstrument,))
            plt.title('autoTune: %s' % (self.__repr__(),), fontsize=14)

        self.goodpeaks = goodpeaks
        self.data['tunex'] = x
        self.data['tuney'] = y
        return goodpeaks

    def autoTuneInit(sd, scanjob, mode='center'):
        stepdata = scanjob.get('stepdata', None)
        sweepdata = scanjob['sweepdata']

        stepparam = sweepdata['param']
        sweepparam = sweepdata['param']

        # set sweep to center
        gates = sd.station.gates
        gates.set(
            sweepparam, (sweepdata['start'] + sweepdata['end']) / 2)
        if stepdata is not None:
            if mode == 'end':
                # set sweep to center
                gates.set(stepparam, (stepdata['end']))
            elif mode == 'start':
                # set sweep to center
                gates.set(stepparam, (stepdata['start']))
            else:
                # set sweep to center
                gates.set(stepparam, (stepdata['start'] + stepdata['end']) / 2)

    def _measurement_post_processing(self, dataset):

        if self.boxcar_filter_kernel_size > 1:
            process_dataarray(dataset, dataset.default_parameter_name(), None, partial(
                qtt.algorithms.generic.boxcar_filter, kernel_size=(self.boxcar_filter_kernel_size,)))

        return dataset

    def fastTune(self, Naverage=90, sweeprange=79, period=1e-3, location=None,
                 fig=201, sleeptime=2, delete=True, add_slopes=False, invert=False, verbose=1):
        """ Fast tuning of the sensing dot plunger.

        If the sensing dot object is initialized with a virtual gates object the virtual plunger will be used
        for the sweep.

        Args:
            Naverage (int): number of averages
            scanrange (float): Range to be used for scanning
            period (float): Period to be used in the scan sweep
            fig (int or None): window for plotting results

        Returns:
            plungervalue (float): value of plunger
            alldata (dataset): measured data
        """

        if self.minstrument is not None:
            instrument = self.minstrument[0]
            channel = self.minstrument[1]
            if not isinstance(channel, list):
                channel = [channel]

            scanjob = qtt.measurements.scans.scanjob_t(
                {'Naverage': Naverage, })
            if self.virt_gates is not None:
                vsensorgate = self.virt_gates.vgates()[self.virt_gates.pgates().index(self.gg[1])]
                scanjob['sweepdata'] = qtt.measurements.scans.create_vectorscan(
                    self.virt_gates.parameters[vsensorgate], g_range=sweeprange, remove_slow_gates=True,
                    station=self.station)
                scanjob['sweepdata']['paramname'] = vsensorgate
            else:
                gate = self.gg[1]
                cc = self.station.gates.get(gate)
                scanjob['sweepdata'] = {'param': gate, 'start': cc -
                                                                sweeprange / 2, 'end': cc + sweeprange / 2, 'step': 4}

            scanjob['sweepdata']['period'] = period
            scanjob['minstrument'] = channel
            scanjob['minstrumenthandle'] = instrument
            scanjob['wait_time_startscan'] = sleeptime
            scanjob['dataset_label'] = 'sensingdot_fast_tune'
            alldata = qtt.measurements.scans.scan1Dfast(self.station, scanjob)
        else:
            raise Exception('legacy code, please do not use')

        alldata.add_metadata({'scanjob': scanjob, 'scantype': 'fastTune'})
        alldata.add_metadata({'snapshot': self.station.snapshot()})

        alldata = self._measurement_post_processing(alldata)

        alldata.write(write_metadata=True)

        goodpeaks = self._process_scan(alldata, useslopes=add_slopes, fig=fig, invert=invert)

        self._select_results(goodpeaks)

        if self.verbose:
            print('sensingdot_t: autotune complete: value %.1f [mV]' % self.sdval[1])

        return self.sdval[1], alldata


class VectorParameter(qcodes.instrument.parameter.Parameter):
    """Create parameter which controls linear combinations.

    Attributes:
        name (str): the name given to the new parameter
        comb_map (list): tuples with in the first entry a parameter and in the
                 second a coefficient
        coeffs_sum (float): the sum of all the coefficients
    """

    def __init__(self, name, comb_map, **kwargs):
        """Initialize a linear combination parameter."""
        super().__init__(name, **kwargs)
        self.comb_map = comb_map
        self.unit = self.comb_map[0][0].unit
        self.coeffs_sum = sum([np.abs(coeff)
                               for (param, coeff) in self.comb_map])

    def get_raw(self):
        """Return the value of this parameter."""
        value = sum([coeff * param.get() for (param, coeff) in self.comb_map])
        return value

    def set_raw(self, value):
        """Set the parameter to value.

        Note: the set is not unique, i.e. the result of this method depends on
        the previous value of this parameter.

        Args:
            value (float): the value to set the parameter to.
        """
        val_diff = value - self.get()
        for (param, coeff) in self.comb_map:
            param.set(param.get() + coeff * val_diff / self.coeffs_sum)


# %%


class MultiParameter(qcodes.instrument.parameter.Parameter):
    """ Create a parameter which is a combination of multiple other parameters.

    Attributes:
        name (str): name for the parameter
        params (list): the parameters to combine
    """

    def __init__(self, name, params, label=None, unit=None, **kwargs):
        super().__init__(name, **kwargs)
        self.params = params
        self.vals = qcodes.utils.validators.Anything()
        self._instrument = 'dummy'
        if label is None:
            self.label = self.name
        if unit is None:
            self.unit = 'a.u.'
        else:
            self.unit = unit
        self.vals = None

    def get_raw(self):
        values = []
        for p in self.params:
            values.append(p.get())
        return values

    def set_raw(self, values):
        for idp, p in enumerate(self.params):
            p.set(values[idp])


class CombiParameter(qcodes.instrument.parameter.Parameter):
    """ Create a parameter which is a combination of multiple other parameters, which are always set to the same value.

    The `get` function returns the mean of the individual parameters.

    Attributes:
        name (str): name for the parameter
        params (list): the parameters to combine
    """

    def __init__(self, name, params, label=None, unit='a.u.', **kwargs):
        super().__init__(name, vals=qcodes.utils.validators.Anything(), unit=unit, **kwargs)
        self.params = params
        if label is None:
            self.label = self.name
        else:
            self.label = label

    def get_raw(self):
        values = []
        for p in self.params:
            values.append(p.get())
        return np.mean(values)

    def set_raw(self, value):
        for idp, p in enumerate(self.params):
            p.set(value)
