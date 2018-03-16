"""
Contains code for various structures

"""
import numpy as np
import time
import matplotlib.pyplot as plt
import warnings

import qcodes

import qtt
import qtt.measurements.scans
from qtt.algorithms.coulomb import peakdataOrientation, coulombPeaks, findSensingDotPosition
from qtt.tools import freezeclass

#%%


@freezeclass
class twodot_t(dict):

    def __init(self, gates, name=None):
        """ Class to represent a double quantum dot """
        self['gates'] = gates

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
            #print('deepcopy %s' % k)
            if k not in ['station']:
                d[k] = copy.deepcopy(v)
        return d


@freezeclass
class onedot_t(dict):
    """ Class representing a single quantum dot """

    def __init__(self, gates, name=None, data=None, station=None, transport_instrument=None):
        """ Class to represent a single quantum dot

        Args:
            gates (list): names of gates to use for left barrier, plunger and right barrier
            name (str): for for the object
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
            #print('deepcopy %s' % k)
            if k not in ['station']:
                d[k] = copy.deepcopy(v)
        return d


def test_spin_structures(verbose=0):
    import pickle
    import json
    # station=qcodes.Station()
    o = onedot_t('dot1', ['L', 'P1', 'D1'], station=None)
    if verbose:
        print('test_spin_structures: %s' % (o,))
    _ = pickle.dumps(o)
    x=json.dumps(o)
    if verbose:
        print('test_spin_structures: %s' % (x,))


if __name__ == '__main__':
    test_spin_structures()

#%%

def _scanlabel(ds):
    """ Helper function """
    a=ds.default_parameter_array()
    lx = a.set_arrays[0].label
    unit=a.set_arrays[0].unit
    if unit:
        lx += '[' + unit +']'
    return lx


@freezeclass
class sensingdot_t:

    def __init__(self, gate_names, gate_values=None, station=None, index=None, minstrument=None, fpga_ch=None):
        """ Class representing a sensing dot 

        We assume the sensing dot can be controlled by two barrier gates and a single plunger gate
        An instrument to measure the current through the dot 
        Args:
            gate_names (list): gates to be used
            gate_values (array or None): values to be set on the gates
            station (Qcodes station)
            minstrument (tuple): measurement instrument to use. tuple of instrument and channel index

            index (None or int): deprecated
            fpga_ch (deprecated, int): index of FPGA channel to use for readout
        """
        self.verbose = 1
        self.gg = gate_names
        if gate_values is None:
            gate_values = [station.gates.get(g) for g in self.gg]
        self.sdval = gate_values
        self.targetvalue = np.NaN
        
        self._detune_axis=np.array([1,-1])
        self._detune_axis=self._detune_axis/np.linalg.norm(self._detune_axis)
        self._debug = {} # store debug data
        self.goodpeaks = None
        self.station = station
        self.index = index
        self.minstrument = minstrument
        self.instrument = 'keithley%d' % index

        self.data = {}

        if fpga_ch is None:
            self.fpga_ch = None 
        else:
            raise Exception('do not use fpga_ch argument, use minstrument instead')
            self.fpga_ch = fpga_ch

        # values for measurement
        if index is not None:
            self.valuefunc = station.components[
                'keithley%d' % index].amplitude.get

    def __repr__(self):
        return 'sd gates: %s, %s, %s' % (self.gg[0], self.gg[1], self.gg[2])

    def __getstate__(self):
        """ Override to make the object pickable."""
        print('sensingdot_t: __getstate__')
        # d=super().__getstate__()
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
        if not sdval is None:
            self.sdval = sdval
        gg = self.gg
        for ii in [0, 2]:
            gates.set(gg[ii], self.sdval[ii])
        if setPlunger:
            ii = 1
            gates.set(gg[ii], self.sdval[ii])

    def tunegate(self):
        """Return the gate used for tuning."""
        return self.gg[1]

    def value(self):
        """Return current through sensing dot """
        if self.valuefunc is not None:
            return self.valuefunc()
        raise Exception(
            'value function is not defined for this sensing dot object')

    def scan1D(sd, outputdir=None, step=-2., max_wait_time=.75, scanrange=300):
        """Make 1D-scan of the sensing dot."""
        print('### sensing dot scan')
        keithleyidx = [sd.index]
        gg = sd.gg
        sdval = sd.sdval
        gates = sd.station.gates

        for ii in [0, 2]:
            gates.set(gg[ii], sdval[ii])

        startval = sdval[1] + scanrange
        startval = np.minimum(startval, 300)
        endval = sdval[1] - scanrange
        endval = np.maximum(endval, -700)

        wait_time = .8
        try:
            wait_time = sd.station.gate_settle(gg[1])
        except:
            pass
        wait_time = np.minimum(wait_time, max_wait_time)

        scanjob1 = qtt.measurements.scans.scanjob_t()
        scanjob1['sweepdata'] = dict(
            {'param': gg[1], 'start': startval, 'end': endval, 'step': step, 'wait_time': wait_time})
        scanjob1['wait_time_startscan'] = .2+3*wait_time
        scanjob1['minstrument'] = keithleyidx
        scanjob1['compensateGates'] = []
        scanjob1['gate_values_corners'] = [[]]

        print('sensingdot_t: scan1D: gate %s, wait_time %.3f' %
              (sd.gg[1], wait_time))

        alldata = qtt.measurements.scans.scan1D(sd.station, scanjob=scanjob1)

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
        
        gates=sd.station.gates
        gv0=gates.allvalues()
        detunings=stepsize * np.arange(-(nsteps-1)/2, nsteps/2)
        dd=[]
        pp=[]
        for ii, dt in enumerate(detunings):
            if verbose:
                print('detuning_scan: iteration %d: detuning %.3f' % (ii, dt) )
            gates.resetgates(sd.gate_names(), gv0, verbose=0)
            sd.detune(dt)
            p, result=sd.fastTune(fig=None, verbose=verbose>=2)
            dd.append(result)
            pp.append(sd.goodpeaks[0])
        gates.resetgates(sd.gate_names(), gv0, verbose=0)
    
        scores=[ p['score'] for p in pp]
        bestidx = np.argmax(scores)
        optimal=[detunings[bestidx], pp[bestidx]['x']]
        if verbose:
                print('detuning_scan: best %d: detuning %.3f' % (bestidx, optimal[0]) )
        results={'detunings': detunings, 'scores': scores, 'bestpeak': pp[bestidx]}
    
        if fig:
            plt.figure(fig); plt.clf()
            for ii in range(len(detunings)):
                ds=dd[ii]
                p=pp[ii]
                y=ds.default_parameter_array()
                x=y.set_arrays[0]
                plt.plot(x, y, '-', label='scan %d: score %.3f' % (ii, p['score']) )
                xx=np.array([p['x'], p['y'] ]).reshape( (2,1))
                qtt.pgeometry.plotLabels(xx, [scores[ii] ])
                plt.title('Detuning scans for %s' % sd.__repr__() )
                plt.xlabel(_scanlabel(result))
            plt.legend(numpoints=1)
            if verbose>=2:
                plt.figure(fig+1); plt.clf()
                plt.plot(detunings, scores, '.', label='Peak scores')
                plt.xlabel('Detuning [mV?]')
                plt.ylabel('Score')
                plt.title('Best peak scores for different detunings')
    
        return optimal, results

    def detune(self, value):
        """ Detune the sensing dot by the specified amount """
        
        gl=getattr(self.station.gates, self.gg[0])
        gr=getattr(self.station.gates, self.gg[2])
        gl.increment(self._detune_axis[0]*value)
        gr.increment(self._detune_axis[1]*value)
        
    def scan2D(sd, ds=90, stepsize=4, fig=None, verbose=1):
        """Make 2D-scan of the sensing dot."""
        #keithleyidx = [sd.index]
        
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

        if sd.verbose>=1:
            print('sensing dot %s: performing barrier-barrier scan' % (sd,))
            if verbose>=2:
                print(scanjob)
        alldata = qtt.measurements.scans.scan2D(sd.station, scanjob, verbose=verbose>=2)

        sd.station.gates.resetgates(gv, gv, verbose=0)
        
        if fig is not None:
            qtt.measurements.scans.plotData(alldata, fig=fig)
        return alldata

    def autoTune(sd, scanjob=None, fig=200, outputdir=None, step=-2.,
                 max_wait_time=1., scanrange=300, add_slopes=False):
        if not scanjob is None:
            sd.autoTuneInit(scanjob)
        alldata = sd.scan1D(outputdir=outputdir, step=step,
                            scanrange=scanrange, max_wait_time=max_wait_time)

        goodpeaks = sd._process_scan(alldata, useslopes=add_slopes, fig=fig)

        if len(goodpeaks) > 0:
            sd.sdval[1] = goodpeaks[0]['xhalfl']
            sd.targetvalue = goodpeaks[0]['yhalfl']
        else:
            print('autoTune: could not find good peak')

        if sd.verbose:
            print(
                'sensingdot_t: autotune complete: value %.1f [mV]' % sd.sdval[1])
        return sd.sdval[1], alldata

    def _process_scan(self, alldata, useslopes=True, fig=None, verbose=0):
        """ Determine peaks in 1D scan """
        istep = float(np.abs(alldata.metadata['scanjob']['sweepdata']['step']))
        x, y = qtt.data.dataset1Ddata(alldata)
        x, y = peakdataOrientation(x, y)

        if useslopes:
            goodpeaks = findSensingDotPosition(
                x, y, useslopes=useslopes, fig=fig, verbose=verbose, istep=istep)
        else:

            goodpeaks = coulombPeaks(
                x, y, verbose=verbose, fig=fig, plothalf=True, istep=istep)
        if fig is not None:
            plt.xlabel('%s' % (self.tunegate(), ))
            plt.ylabel('%s' % (self.minstrument, ))
            plt.title('autoTune: %s' % (self.__repr__(), ), fontsize=14)

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
        if not stepdata is None:
            if mode == 'end':
                # set sweep to center
                gates.set(stepparam, (stepdata['end']))
            elif mode == 'start':
                # set sweep to center
                gates.set(stepparam, (stepdata['start']))
            else:
                # set sweep to center
                gates.set(stepparam, (stepdata['start'] + stepdata['end']) / 2)

    def fastTune(self, Naverage=50, sweeprange=79, period=.5e-3, location=None,
                 fig=201, sleeptime=2, delete=True, add_slopes=False, verbose=1):
        """Fast tuning of the sensing dot plunger.

        Args:
            fig (int or None): window for plotting results
            Naverage (int): number of averages
            
        Returns:
            plungervalue (float): value of plunger
            alldata (dataset): measured data
        """

        if self.minstrument is not None:
            instrument = self.minstrument[0]
            channel = self.minstrument[1]
            gate = self.gg[1]
            sdplg = getattr(self.station.gates, gate)
            cc = self.station.gates.get(gate)
            scanjob = qtt.measurements.scans.scanjob_t(
                {'Naverage': Naverage, })
            scanjob['sweepdata'] = {'param':  gate, 'start': cc -
                                    sweeprange / 2, 'end': cc + sweeprange / 2, 'step': 4}
            scanjob['minstrument'] = [channel]
            scanjob['minstrumenthandle'] = instrument
            scanjob['wait_time_startscan'] = sleeptime

            alldata = qtt.measurements.scans.scan1Dfast(self.station, scanjob)
        else:
            waveform, sweep_info = self.station.awg.sweep_gate(
                self.gg[1], sweeprange, period, wave_name='fastTune_%s' % self.gg[1], delete=delete)

            # time for AWG signal to reach the sample
            qtt.time.sleep(sleeptime)

            ReadDevice = ['FPGA_ch%d' % self.fpga_ch]
            _, DataRead_ch1, DataRead_ch2 = self.station.fpga.readFPGA(
                Naverage=Naverage, ReadDevice=ReadDevice)

            self.station.awg.stop()

            if self.fpga_ch == 1:
                datr = DataRead_ch1
            else:
                datr = DataRead_ch2
            data = self.station.awg.sweep_process(datr, waveform, Naverage)

            sdplg = getattr(self.station.gates, self.gg[1])
            initval = sdplg.get()
            sweepvalues = sdplg[initval - sweeprange /
                                2:sweeprange / 2 + initval:sweeprange / len(data)]

            alldata = qtt.data.makeDataSet1D(sweepvalues, y=data, location=location, loc_record={
                                             'label': 'sensingdot_t.fastTune'})

        #alldata= qtt.scans.scan1Dfast(self.station, scanjob, location=location)

        alldata.add_metadata({'scanjob': scanjob, 'scantype': 'fastTune'})
        alldata.add_metadata({'snapshot': self.station.snapshot()})

        alldata.write(write_metadata=True)

        goodpeaks = self._process_scan(alldata, useslopes=add_slopes, fig=fig)

        if len(goodpeaks) > 0:
            self.sdval[1] = goodpeaks[0]['xhalfl']
            self.targetvalue = goodpeaks[0]['yhalfl']
        else:
            print('autoTune: could not find good peak, may need to adjust mirrorfactor')

        if self.verbose:
            print(
                'sensingdot_t: autotune complete: value %.1f [mV]' % self.sdval[1])

        return self.sdval[1], alldata

    def fastTune_virt(self, virt_gates, Naverage=50, sweeprange=79, period=.5e-3, location=None,
                 fig=201, sleeptime=2, delete=True, add_slopes=False):
        """Fast tuning of the sensing dot using the virtual plunger.

        Args:
            virt_gates (virtual_gates instrument): virtual gate object used for the scan
            fig (int or None): window for plotting results
            Naverage (int): number of averages
            
        Returns:
            plungervalue (float): value of plunger
            alldata (dataset): measured data
        """

        if self.minstrument is not None:
            instrument = self.minstrument[0]
            channel = self.minstrument[1]
            vsensorgate=virt_gates.vgates()[virt_gates.pgates().index(self.gg[1])] #name of the virtual plunger
            scanjob = qtt.measurements.scans.scanjob_t(
                {'Naverage': Naverage, })
            scanjob['sweepdata'] = qtt.measurements.scans.create_vectorscan(virt_gates.parameters[vsensorgate], g_range=sweeprange, remove_slow_gates=True, station=self.station)
            scanjob['sweepdata']['paramname'] = vsensorgate
            scanjob['minstrument'] = [channel]
            scanjob['minstrumenthandle'] = instrument
            scanjob['wait_time_startscan'] = sleeptime

            alldata = qtt.measurements.scans.scan1Dfast(self.station, scanjob)


        alldata.add_metadata({'scanjob': scanjob, 'scantype': 'fastTune'})
        alldata.add_metadata({'snapshot': self.station.snapshot()})

        alldata.write(write_metadata=True)

        goodpeaks = self._process_scan(alldata, useslopes=add_slopes, fig=fig)

        if len(goodpeaks) > 0:
            self.sdval[1] = goodpeaks[0]['xhalfl']
            self.targetvalue = goodpeaks[0]['yhalfl']
        else:
            print('autoTune: could not find good peak, may need to adjust mirrorfactor')

        if self.verbose:
            print(
                'sensingdot_t: autotune complete: value %.1f [mV]' % self.sdval[1])

        return self.sdval[1], alldata    

#%%
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
        self.name = name
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

#%%


class MultiParameter(qcodes.instrument.parameter.Parameter):
    """ Create a parameter which is a combination of multiple other parameters.

    Attributes:
        name (str): name for the parameter
        params (list): the parameters to combine
    """

    def __init__(self, name, params, label=None, unit=None, **kwargs):
        super().__init__(name, **kwargs)
        self.name = name
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

        # for backwards compatibility
        self.has_get = True
        self.has_set = True

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
        super().__init__(name, vals = qcodes.utils.validators.Anything(), unit = unit, **kwargs)
        self.params = params
        if label is None:
            self.label = self.name
        else:
            self.label = label
        
        # for backwards compatibility
        self.has_get = True
        self.has_set = True

    def get_raw(self):
        values = []
        for p in self.params:
            values.append(p.get())
        return np.mean(values)

    def set_raw(self, value):
        for idp, p in enumerate(self.params):
            p.set(value)


def test_multi_parameter():
    p = qcodes.Parameter('p', set_cmd=None)
    q = qcodes.Parameter('q', set_cmd=None)
    mp = MultiParameter('multi_param', [p, q])
    mp.set([1, 2])
    _ = mp.get()
    mp = CombiParameter('multi_param', [p, q])
    mp.set([1, 3])
    v = mp.get()
    assert(v == 2)


if __name__ == '__main__':
    import qcodes
    test_multi_parameter()
