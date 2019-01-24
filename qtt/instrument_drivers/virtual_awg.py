# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 13:04:09 2016

@author: diepencjv
"""

#%%
import numpy as np
import scipy.signal
import logging
import warnings

import qcodes
from qcodes import Instrument
from qcodes.plots.pyqtgraph import QtPlot
from qcodes import DataArray
import qtt
import qtt.utilities.tools

logger = logging.getLogger(__name__)
#%%


class virtual_awg(Instrument):
    """ 
    
    Attributes:
        _awgs (list): handles to instruments
        awg_map (dict)
        hardware (Instrument): contains AWG to plunger values
        corr (float): unknown
        delay_FPGA (float): time delay of signals going through fridge
        
    """
    def __init__(self, name, instruments=[], awg_map=None, hardware=None, verbose=1, **kwargs):
        super().__init__(name, **kwargs)
        logger.info('initialize virtual_awg %s' % name)
        self._awgs = instruments
        self.awg_map = awg_map
        self.hardware = hardware
        self.verbose = verbose
        self.delay_FPGA = 2.0e-6  # should depend on filterboxes
        self.corr = .0 # legacy code, specific for FPGA board not used any more
        self.maxdatapts = 16e6  # This used to be set to the fpga maximum, but that maximum should not be handled here

        self.awg_seq = None
        if len(self._awgs) == 0 and self.verbose:
            print('no physical AWGs connected')
        elif len(self._awgs) == 1:
            self.awg_cont = self._awgs[0]
            self.awg_cont.set('run_mode', 'CONT')
        elif len(self._awgs) == 2 and 'awg_mk' in self.awg_map:
            self.awg_cont = self._awgs[self.awg_map['awg_mk'][0]]
            self.awg_cont.set('run_mode', 'CONT')
            self.awg_seq = self._awgs[(self.awg_map['awg_mk'][0] + 1) % 2]
            
            self._set_seq_mode(self.awg_seq)
            self.delay_AWG = self.hardware.parameters['delay_AWG'].get()
        else:
            raise Exception(
                'Configuration of AWGs not supported by virtual_awg instrument')

        self.AWG_clock = 1e8
        self.ch_amp = 4.0
        for awg in self._awgs:
            awg.set('clock_freq', self.AWG_clock)
            awg.delete_all_waveforms_from_list()
            for i in range(1, 5):
                awg.set('ch%s_amp' % i, self.ch_amp)

    def _set_seq_mode(self, a):
        a.set('run_mode', 'SEQ')
        a.sequence_length.set(1)
        a.set_sqel_trigger_wait(1, 0)
        
    def get_idn(self):
        ''' Overrule because the default VISA command does not work '''
        IDN = {'vendor': 'QuTech', 'model': 'virtual_awg',
               'serial': None, 'firmware': None}
        return IDN

    def awg_gate(self, gate):
        """ Return true of the gate can be controlled by the awg
        
        Args:
            gate ()
        """
        if gate is None:
            return False
        
        if isinstance(gate, dict):
            # vector scan, assume we can do it fast if all components are fast
            return np.all([self.awg_gate(g) for g in gate])
        if self.awg_map is None:
            return False
        
        if gate in self.awg_map:
            return True
        else:
            return False
        
    def stop(self, verbose=0):
        ''' Stops all AWGs and turns of all channels '''
        for awg in self._awgs:
            awg.stop()
            for i in range(1, 5):
                awg.set('ch%d_state' % i, 0)

        if verbose:
            print('Stopped AWGs')

    def sweep_init(self, waveforms, period=1e-3, delete=True, samp_freq=None):
        ''' Send waveform(s) to gate(s)

        Arguments:
            waveforms (dict): the waveforms with the gates as keys
            period (float): period of the waveform in seconds

        Returns:
            sweep_info (dict): the keys are tuples of the awgs and channels to activate

        Example:
        --------
        >> sweep_info = sweep_init(waveforms)
        '''
        sweepgates = [g for g in waveforms]

        if delete:
            for awg in self._awgs:
                awg.delete_all_waveforms_from_list()

        awgs = [self._awgs[self.awg_map[g][0]] for g in sweepgates]
        if 'fpga_mk' in self.awg_map:
            marker_info = self.awg_map['fpga_mk']
            marker_delay = self.delay_FPGA
            marker_name = 'fpga_mk'
        elif 'm4i_mk' in self.awg_map:
            marker_info = self.awg_map['m4i_mk']
            if samp_freq is not None:
                pretrigger_period = 16 / samp_freq
            else:
                pretrigger_period = 0
            marker_delay = self.delay_FPGA + pretrigger_period
            marker_name = 'm4i_mk'

        awgs.append(self._awgs[marker_info[0]])

        sweep_info = dict()
        wave_len = len(waveforms[sweepgates[0]]['wave'])
        for g in sweepgates:
            sweep_info[self.awg_map[g]] = dict()
            sweep_info[self.awg_map[g]]['waveform'] = waveforms[g]['wave']
            sweep_info[self.awg_map[g]]['marker1'] = np.zeros(wave_len)
            sweep_info[self.awg_map[g]]['marker2'] = np.zeros(wave_len)
            if 'name' in waveforms[g]:
                sweep_info[self.awg_map[g]]['name'] = waveforms[g]['name']
            else:
                sweep_info[self.awg_map[g]]['name'] = 'waveform_%s' % g
            if marker_info[:2] == self.awg_map[g]:
                sweep_info[marker_info[:2]]['delay'] = marker_delay

        # marker points
        marker_points = np.zeros(wave_len)
        marker_points[int(marker_delay * self.AWG_clock):(int(marker_delay * self.AWG_clock) + wave_len // 20)] = 1.0

        if marker_info[:2] not in sweep_info:
            sweep_info[marker_info[:2]] = dict()
            sweep_info[marker_info[:2]]['waveform'] = np.zeros(wave_len)
            sweep_info[marker_info[:2]]['marker1'] = np.zeros(wave_len)
            sweep_info[marker_info[:2]]['marker2'] = np.zeros(wave_len)
            for g in sweepgates:
                marker_name += '_%s' % g
            sweep_info[marker_info[:2]]['name'] = marker_name
            sweep_info[marker_info[:2]]['delay'] = marker_delay

        sweep_info[marker_info[:2]]['marker%d' % marker_info[2]] = marker_points
        self._awgs[marker_info[0]].set(
            'ch%i_m%i_low' % (marker_info[1], marker_info[2]), 0)
        self._awgs[marker_info[0]].set(
            'ch%i_m%i_high' % (marker_info[1], marker_info[2]), 2.6)

        # awg marker
        if getattr(self, 'awg_seq', None) is not None:
            awg_info = self.awg_map['awg_mk']
            if awg_info[:2] not in sweep_info:
                awgs.append(self._awgs[awg_info[0]])
                sweep_info[awg_info[:2]] = dict()
                sweep_info[awg_info[:2]]['waveform'] = np.zeros(wave_len)
                sweep_info[awg_info[:2]]['marker1'] = np.zeros(wave_len)
                sweep_info[awg_info[:2]]['marker2'] = np.zeros(wave_len)
                sweep_info[awg_info[:2]]['name'] = 'awg_mk'

            awg_marker = np.zeros(wave_len)
            awg_marker[0:wave_len // 20] = 1
            awg_marker = np.roll(
                awg_marker, wave_len - int(self.delay_AWG * self.AWG_clock))
            sweep_info[awg_info[:2]]['marker%d' %
                                     self.awg_map['awg_mk'][2]] = awg_marker
            self._awgs[awg_info[0]].set(
                'ch%i_m%i_low' % (awg_info[1], awg_info[2]), 0)
            self._awgs[awg_info[0]].set(
                'ch%i_m%i_high' % (awg_info[1], awg_info[2]), 2.6)

        # send waveforms
        if delete:
            for sweep in sweep_info:
                try:
                    self._awgs[sweep[0]].send_waveform_to_list(sweep_info[sweep]['waveform'], sweep_info[
                        sweep]['marker1'], sweep_info[sweep]['marker2'], sweep_info[sweep]['name'])
                except Exception as ex:
                    print(ex)
                    print('sweep_info[sweep][waveform] %s' % (sweep_info[sweep]['waveform'].shape,))
                    print('sweep_info[sweep][marker1] %s' % (sweep_info[sweep]['marker1'].shape,))
                    print('sweep_info[sweep][marker2] %s' % (sweep_info[sweep]['marker2'].shape,))

        return sweep_info

    def sweep_run(self, sweep_info):
        ''' Activate AWG(s) and channel(s) for the sweep(s).

        Arguments:
            sweep_info (dict): the keys are tuples of the awgs and channels to activate
        '''
        for sweep in sweep_info:
            if hasattr(self, 'awg_seq') and self._awgs[sweep[0]] == self.awg_seq:
                self._awgs[sweep[0]].set_sqel_waveform(
                    sweep_info[sweep]['name'], sweep[1], 1)
                self._awgs[sweep[0]].set_sqel_loopcnt_to_inf(1)
                self._awgs[sweep[0]].set_sqel_event_jump_target_index(
                    sweep[1], 1)
                self._awgs[sweep[0]].set_sqel_event_jump_type(1, 'IND')
            else:
                self._awgs[sweep[0]].set(
                    'ch%i_waveform' % sweep[1], sweep_info[sweep]['name'])

        for sweep in sweep_info:
            self._awgs[sweep[0]].set('ch%i_state' % sweep[1], 1)

        awgnrs = set([sweep[0] for sweep in sweep_info])
        for nr in awgnrs:
            self._awgs[nr].run()

    def make_sawtooth(self, sweeprange, period, width=.95, repetitionnr=1, start_zero=False):
        '''Make a sawtooth with a decline width determined by width. Not yet scaled with
        awg_to_plunger value.

        Arguments:
            sweeprange (float): the range of voltages to sweep over
            period (float): the period of the triangular signal

        Returns:
            wave_raw (array): raw data which represents the waveform
        '''
        samplerate = 1. / self.AWG_clock
        tt = np.arange(0, period * repetitionnr + samplerate, samplerate)
        v_wave = float(sweeprange / ((self.ch_amp / 2.0)))
        wave_raw = (v_wave / 2) * scipy.signal.sawtooth(2 * np.pi * tt / period, width=width)
#        idx_zero = np.argmin(np.abs(wave_raw))
#        wave_raw = np.roll(wave_raw, wave_raw.size-idx_zero)
        if start_zero:
            o=int((wave_raw.size)*(1-width)/2)
            wave_raw = np.roll(wave_raw, o)

        return wave_raw
    
    def make_pulses(self, voltages, waittimes, reps=1, filtercutoff=None, mvrange=None):
        """Make a pulse sequence with custom voltage levels and wait times at each level.
        
        Arguments:
            voltages (list of floats): voltage levels to be applied in the sequence
            waittimes (list of floats): duration of each pulse in the sequence
            reps (int): number of times to repeat the pulse sequence in the waveform
            filtercutoff (float): cutoff frequency of a 1st order butterworth filter to make the pulse steps smoother 
            
        Returns:
            wave_raw (array): raw data which represents the waveform
        """
        if len(waittimes) != len(voltages):
            raise Exception('Number of voltage levels must be equal to the number of wait times')
        samples = [int(x * self.AWG_clock) for x in waittimes]
        if mvrange is None:
            mvrange = [max(voltages), min(voltages)]
        v_wave = float((mvrange[0] - mvrange[1]) / self.ch_amp)
        v_prop = [2 * ((x - mvrange[1]) / (mvrange[0] - mvrange[1])) - 1 for x in voltages]
        wave_raw = np.concatenate([x * v_wave * np.ones(y) for x, y in zip(v_prop, samples)])
        if filtercutoff is not None:
            b,a = scipy.signal.butter(1,0.5*filtercutoff/self.AWG_clock, btype='low', analog=False, output='ba')
            wave_raw = scipy.signal.filtfilt(b,a,wave_raw)
        wave_raw = np.tile(wave_raw, reps)
            
        return wave_raw

    def check_frequency_waveform(self, period, width):
        """ Check whether a sawtooth waveform with specified period can be generated """
        old_sr = self.AWG_clock
        new_sr = 5 / (period * (1 - width))
        if (new_sr) > old_sr:
            warnings.warn('awg sampling frequency %.1f MHz is too low for signal requested (sr %.1f [MHz], period %.1f [ms])' % (old_sr / 1e6, new_sr / 1e6, 1e3 * period), UserWarning)
        return new_sr

    def sweep_gate(self, gate, sweeprange, period, width=.95, wave_name=None, delete=True, samp_freq=None):
        ''' Send a sawtooth signal with the AWG to a gate to sweep. Also
        send a marker to the measurement instrument.

        Args:
            gate (string): the name of the gate to sweep
            sweeprange (float): the range of voltages to sweep over
            period (float): the period of the triangular signal

        Returns:
            waveform (dict): The waveform being send with the AWG.
            sweep_info (dict): the keys are tuples of the awgs and channels to activate

        Example:
            >>> waveform, sweep_info = sweep_gate('P1',sweeprange=60,period=1e-3)
        '''

        self.check_frequency_waveform(period, width)
        self.check_amplitude(gate, sweeprange)

        start_zero=True
        waveform = dict()
        wave_raw = self.make_sawtooth(sweeprange, period, width, start_zero=start_zero)
        awg_to_plunger = self.hardware.parameters['awg_to_%s' % gate].get()
        wave = wave_raw / awg_to_plunger
        waveform[gate] = dict()
        waveform[gate]['wave'] = wave
        if wave_name is None:
            waveform[gate]['name'] = 'sweep_%s' % gate
        else:
            waveform[gate]['name'] = wave_name
        sweep_info = self.sweep_init(waveform, period, delete, samp_freq=samp_freq)
        self.sweep_run(sweep_info)
        waveform['width'] = width
        waveform['start_zero']=start_zero
        waveform['sweeprange'] = sweeprange
        waveform['samplerate'] = 1 / self.AWG_clock
        waveform['period'] = period
        for channels in sweep_info:
            if 'delay' in sweep_info[channels]:
                waveform['markerdelay'] = sweep_info[channels]['delay']

        return waveform, sweep_info

    def sweep_gate_virt(self, gate_comb, sweeprange, period, width=.95, delete=True, samp_freq=None):
        ''' Send a sawtooth signal with the AWG to a linear combination of 
        gates to sweep. Also send a marker to the measurement instrument.

        Arguments:
            gate_comb (dict): the gates to sweep and the coefficients as values
            sweeprange (float): the range of voltages to sweep over
            period (float): the period of the triangular signal

        Returns:
            waveform (dict): The waveform being send with the AWG.
            sweep_info (dict): the keys are tuples of the awgs and channels to activate
        '''

        self.check_frequency_waveform(period, width)

        waveform = dict()
        for g in gate_comb:
            self.check_amplitude(g, gate_comb[g] * sweeprange)
        for g in gate_comb:
            wave_raw = self.make_sawtooth(sweeprange, period, width)
            awg_to_plunger = self.hardware.parameters['awg_to_%s' % g].get()
            wave = wave_raw * gate_comb[g] / awg_to_plunger
            waveform[g] = dict()
            waveform[g]['wave'] = wave
            waveform[g]['name'] = 'sweep_%s' % g

        sweep_info = self.sweep_init(waveform, period, delete, samp_freq=samp_freq)
        self.sweep_run(sweep_info)
        waveform['width'] = width
        waveform['sweeprange'] = sweeprange
        waveform['samplerate'] = 1 / self.AWG_clock
        waveform['period'] = period
        for channels in sweep_info:
            if 'delay' in sweep_info[channels]:
                waveform['markerdelay'] = sweep_info[channels]['delay']

        return waveform, sweep_info

    def sweepandpulse_gate(self, sweepdata, pulsedata, wave_name=None, delete=True, shift_zero=True):
        ''' Makes and outputs a waveform which overlays a sawtooth signal to sweep 
        a gate, with a pulse sequence. A marker is sent to the measurement instrument 
        at the start of the waveform.
        IMPORTANT: The function offsets the voltages values so that the last point is 0 V on all gates (i.e. it centers the pulse sequence on the last point)

        Args:
            sweepdata (dict): inputs for the sawtooth (gate, sweeprange, period, width). 
                    See sweep_gate for more info.
            pulsedata (dict): inputs for the pulse sequence (gate_voltages, waittimes).
                    See pulse_gates for more info.

        Returns:
            waveform (dict): The waveform being sent with the AWG.
            sweep_info (dict): the keys are tuples of the awgs and channels to activate
        '''

        sweepgate = sweepdata['gate']
        sweeprange = sweepdata['sweeprange']
        period = sweepdata['period']
        width = sweepdata.get('width',0.95)
        
        gate_voltages = pulsedata['gate_voltages'].copy()
        if shift_zero:
            for g in gate_voltages:
                gate_voltages[g] = [x - gate_voltages[g][-1] for x in gate_voltages[g]]
        waittimes = pulsedata['waittimes']
        filtercutoff = pulsedata.get('filtercutoff',None)
        
        pulsesamp = [int(round(x * self.AWG_clock)) for x in waittimes]
        sawsamp = int(round(period * width * self.AWG_clock))
        pulsereps = int(np.ceil(self.AWG_clock * period * width / sum(pulsesamp)))
        allvoltages = np.concatenate([v for v in gate_voltages.values()])
        mvrange = [max(allvoltages), min(allvoltages)]
                
        self.check_frequency_waveform(period, width)

        waveform = dict()
        wave_sweep = self.make_sawtooth(sweeprange, period, width)
        for g in gate_voltages:
            self.check_amplitude(g, sweeprange + (mvrange[0]-mvrange[1]))
        for g in gate_voltages:
            wave_raw = self.make_pulses(gate_voltages[g], waittimes, reps=pulsereps, filtercutoff=filtercutoff, mvrange=mvrange)
            wave_raw = wave_raw[:sawsamp]
            wave_raw = np.pad(wave_raw, (0,len(wave_sweep) - len(wave_raw)), 'edge')
            if sweepgate == g:
                wave_raw += wave_sweep
            awg_to_plunger = self.hardware.parameters['awg_to_%s' % g].get()
            wave = wave_raw / awg_to_plunger
            waveform[g] = dict()
            waveform[g]['wave'] = wave
            if wave_name is None:
                waveform[g]['name'] = 'sweepandpulse_%s' % g
            else:
                waveform[g]['name'] = wave_name
        sweep_info = self.sweep_init(waveform, period, delete)
        self.sweep_run(sweep_info)
        waveform['width'] = width
        waveform['sweeprange'] = sweeprange
        waveform['samplerate'] = 1 / self.AWG_clock
        waveform['period'] = period
        waveform['pulse_voltages'] = gate_voltages
        waveform['pulse_waittimes'] = waittimes
        for channels in sweep_info:
            if 'delay' in sweep_info[channels]:
                waveform['markerdelay'] = sweep_info[channels]['delay']

        return waveform, sweep_info

    def sweep_process(self, data, waveform, Naverage=1, direction='forwards', start_offset=1):
        """ Process the data returned by reading out based on the shape of
        the sawtooth send with the AWG.

        Args:
            data (list or Nxk array): the data (N is the number of samples)
            waveform (dict): contains the wave and the sawtooth width
            Naverage (int): number of times the signal was averaged
            direction (string): option to use backwards signal i.o. forwards

        Returns:
            data_processed (array): The data after dropping part of it.

        Example:
            >> data_processed = sweep_process(data, waveform, 25)
        """
        width = waveform['width']

        if isinstance(data, list):
            data = np.array(data)

        if direction == 'forwards':
            end = int(np.floor(width * data.shape[0] - 1))
            data_processed = data[start_offset:end]
        elif direction == 'backwards':
            begin = int(np.ceil(width * data.shape[0] + 1))
            data_processed = data[begin:]
            data_processed = data_processed[::-1]

        data_processed = np.array(data_processed) / Naverage

        return data_processed

    def sweep_2D(self, samp_freq, sweepgates, sweepranges, resolution, width=.95, comp=None, delete=True):
        ''' Send sawtooth signals to the sweepgates which effectively do a 2D
        scan.

        The first sweepgate is the fast changing gate (on the horizontal axis).
        
        Arguments:
            samp_freq (float): sampling frequency of the measurement instrument in Hertz.
            sweepgates (list): two strings with names of gates to sweep
            sweepranges (list): two floats for sweepranges in milliVolts

        Returns:
            waveform (dict): The waveforms being send with the AWG.
            sweep_info (dict): the keys are tuples of the awgs and channels to activate
        '''
# JP: I think FPGA exceptions should not be handled by awg
#        if resolution[0] * resolution[1] > self.maxdatapts:
#            raise Exception('resolution is set higher than FPGA memory allows')

        if self.corr != 0:
            raise Exception('please do not use the .corr setting any more')
        error_corr = resolution[0] * self.corr
        period_horz = resolution[0] / samp_freq + error_corr
        period_vert = resolution[1] * period_horz

        self.check_frequency_waveform(period_horz, width)
        for g, r in zip(sweepgates, sweepranges):
            self.check_amplitude(g, r)

        waveform = dict()
        # horizontal waveform
        wave_horz_raw = self.make_sawtooth(
            sweepranges[0], period_horz, repetitionnr=resolution[1])
        awg_to_plunger_horz = self.hardware.parameters[
            'awg_to_%s' % sweepgates[0]].get()
        wave_horz = wave_horz_raw / awg_to_plunger_horz
        waveform[sweepgates[0]] = dict()
        waveform[sweepgates[0]]['wave'] = wave_horz
        waveform[sweepgates[0]]['name'] = 'sweep_2D_horz_%s' % sweepgates[0]

        # vertical waveform
        wave_vert_raw = self.make_sawtooth(sweepranges[1], period_vert)
        awg_to_plunger_vert = self.hardware.parameters[
            'awg_to_%s' % sweepgates[1]].get()
        wave_vert = wave_vert_raw / awg_to_plunger_vert
        waveform[sweepgates[1]] = dict()
        waveform[sweepgates[1]]['wave'] = wave_vert
        waveform[sweepgates[1]]['name'] = 'sweep_2D_vert_%s' % sweepgates[1]

        if comp is not None:
            for g in comp:
                if g not in sweepgates:
                    waveform[g] = dict()
                    waveform[g]['wave'] = comp[g]['vert'] * \
                        wave_vert + comp[g]['horz'] * wave_horz
                    waveform[g]['name'] = 'sweep_2D_comp_%s' % g
                else:
                    raise Exception('Can not compensate a sweepgate')

        sweep_info = self.sweep_init(waveform, period=period_vert, delete=delete, samp_freq=samp_freq)
        self.sweep_run(sweep_info)

        waveform['width_horz'] = width
        waveform['sweeprange_horz'] = sweepranges[0]
        waveform['width_vert'] = width
        waveform['sweeprange_vert'] = sweepranges[1]
        waveform['resolution'] = resolution
        waveform['samplerate'] = 1 / self.AWG_clock
        waveform['period'] = period_vert
        waveform['period_horz'] = period_horz
        for channels in sweep_info:
            if 'delay' in sweep_info[channels]:
                waveform['markerdelay'] = sweep_info[channels]['delay']

        return waveform, sweep_info

    def sweep_2D_virt(self, samp_freq, gates_horz, gates_vert, sweepranges, resolution, width=.95, delete=True):
        ''' Send sawtooth signals to the linear combinations of gates set by
        gates_horz and gates_vert which effectively do a 2D scan of two virtual
        gates.

        The horizontal direction is the direction where the AWG signal is changing fastest. It is the first element in the resolution and sweepranges.
        
        Arguments:
            samp_freq (float): sampling frequency of the measurement instrument in Hertz.
            gates_horz (dict): the gates for the horizontal direction and their coefficients
            gates_vert (dict): the gates for the vertical direction and their coefficients
            sweepranges (list): two floats for sweepranges in milliVolts
            resolution (list): two ints for numbers of pixels

        Returns:
            waveform (dict): The waveforms being send with the AWG.
            sweep_info (dict): the keys are tuples of the awgs and channels to activate
        '''
# JP: I think FPGA exceptions should not be handled by awg
#        if resolution[0] * resolution[1] > self.maxdatapts:
#            raise Exception('resolution is set higher than memory allows')

        error_corr = resolution[0] * self.corr
        period_horz = resolution[0] / samp_freq + error_corr
        period_vert = resolution[1] * period_horz

        new_sr = self.check_frequency_waveform(period_horz, width)
        # self.reset_AWG(new_sr)

        waveform = dict()
        # horizontal virtual gate
        for g in gates_horz:
            self.check_amplitude(g, sweepranges[0] * gates_horz[g])
        for g in gates_horz:
            wave_raw = self.make_sawtooth(sweepranges[0], period_horz, repetitionnr=resolution[1])
            awg_to_plunger = self.hardware.parameters['awg_to_%s' % g].get()
            wave = wave_raw * gates_horz[g] / awg_to_plunger
            waveform[g] = dict()
            waveform[g]['wave'] = wave
            waveform[g]['name'] = 'sweep_2D_virt_%s' % g

        # vertical virtual gate
        for g in gates_vert:
            self.check_amplitude(g, sweepranges[1] * gates_vert[g])
        for g in gates_vert:
            wave_raw = self.make_sawtooth(sweepranges[1], period_vert)
            awg_to_plunger = self.hardware.parameters['awg_to_%s' % g].get()
            wave = wave_raw * gates_vert[g] / awg_to_plunger
            if g in waveform:
                waveform[g]['wave'] = waveform[g]['wave'] + wave
            else:
                waveform[g] = dict()
                waveform[g]['wave'] = wave
                waveform[g]['name'] = 'sweep_2D_virt_%s' % g

        # TODO: Implement compensation of sensing dot plunger

        sweep_info = self.sweep_init(waveform, period=period_vert, delete=delete, samp_freq=samp_freq)
        self.sweep_run(sweep_info)

        waveform['width_horz'] = width
        waveform['sweeprange_horz'] = sweepranges[0]
        waveform['width_vert'] = width
        waveform['sweeprange_vert'] = sweepranges[1]
        waveform['resolution'] = resolution
        waveform['samplerate'] = 1 / self.AWG_clock
        waveform['period'] = period_vert
        waveform['period_horz'] = period_horz
        for channels in sweep_info:
            if 'delay' in sweep_info[channels]:
                waveform['markerdelay'] = sweep_info[channels]['delay']

        return waveform, sweep_info

    def sweep_2D_process(self, data, waveform, diff_dir=None):
        ''' Process data from sweep_2D 

        Arguments:
            data (list): the raw measured data
            waveform (dict): The waveforms that was sent with the AWG.

        Returns:
            data_processed (list): the processed data
        '''
        width_horz = waveform['width_horz']
        width_vert = waveform['width_vert']
        resolution = waveform['resolution']

        # split up the fpga data in chunks of horizontal sweeps
        chunks_ch1 = [data[x:x + resolution[0]] for x in range(0, len(data), resolution[0])]
        chunks_ch1 = [chunks_ch1[i][1:int(width_horz * len(chunks_ch1[i]))] for i in range(0, len(chunks_ch1))]
        data_processed = chunks_ch1[:int(width_vert * len(chunks_ch1))]

        if diff_dir is not None:
            data_processed = qtt.utilities.tools.diffImageSmooth(data_processed, dy=diff_dir, sigma=1)

        return data_processed

    def pulse_gates(self, gate_voltages, waittimes, reps=1, filtercutoff=None, reset_to_zero=False, delete=True):
        ''' Send a pulse sequence with the AWG that can span over any gate space.
        Sends a marker to measurement instrument at the start of the sequence.
        Only works with physical gates.

        Arguments:
            gate_voltages (dict): keys are gates to apply the sequence to, and values
            are arrays with the voltage levels to be applied in the sequence
            waittimes (list of floats): duration of each pulse in the sequence
            reset_to_zero (bool): if True, the function offsets the voltages values so that the last point is 0V
                                  on all gates (i.e. it centers the pulse sequence on the last point).

        Returns:
            waveform (dict): The waveform being send with the AWG.
            sweep_info (dict): the keys are tuples of the awgs and channels to activate
        '''

        period = sum(waittimes)
        if reset_to_zero:
            for g in gate_voltages:
                gate_voltages[g] = [x-gate_voltages[g][-1] for x in gate_voltages[g]]
        allvoltages = np.concatenate([v for v in gate_voltages.values()])
        mvrange = [max(allvoltages), min(allvoltages)]
        waveform = dict()
        for g in gate_voltages:
            wave_raw = self.make_pulses(gate_voltages[g], waittimes, reps=reps, filtercutoff=filtercutoff, mvrange=mvrange)
            awg_to_plunger = self.hardware.parameters['awg_to_%s' % g].get()
            wave = wave_raw / awg_to_plunger
            waveform[g] = dict()
            waveform[g]['wave'] = wave
            waveform[g]['name'] = 'pulses_%s' % g

        sweep_info = self.sweep_init(waveform, period, delete)
        self.sweep_run(sweep_info)
        waveform['voltages'] = gate_voltages
        waveform['samplerate'] = 1 / self.AWG_clock
        waveform['waittimes'] = waittimes
        for channels in sweep_info:
            if 'delay' in sweep_info[channels]:
                waveform['markerdelay'] = sweep_info[channels]['delay']

        return waveform, sweep_info
   
    def reset_AWG(self, clock=1e8):
        """ Reset AWG to videomode and scanfast """
        self.AWG_clock = clock
        for a in self._awgs:
            a.clock_freq.set(clock)
            a.trigger_mode.set('CONT')
            a.trigger_source.set('INT')

            for ii in range(1, 5):
                f = getattr(a, 'ch%d_amp' % ii)
                val = f()
                if val != 4.0:
                    warnings.warn('AWG channel %d output not at 4.0 V' % ii)
        if self.awg_seq is not None:
            self._set_seq_mode(self.awg_seq)
            
    def set_amplitude(self, amplitude):
        """ Set the AWG peak-to-peak amplitude for all channels

        Args:
            amplitude (float): peak-to-peak amplitude (V)

        """
        if amplitude < 0.02:
            warnings.warn('Trying to set AWG amplitude too low, setting it to minimum (20mV)')
            amplitude = 0.02
        elif amplitude > 4.5:
            warnings.warn('Trying to set AWG amplitude too high, setting it to maximum (4.5V)')
            amplitude = 4.5

        # tektronics 5014 has precision of 1mV
        self.ch_amp = round(amplitude, 3)
        for awg in self._awgs:
            for i in range(1, 5):
                awg.set('ch%s_amp' % i, self.ch_amp)
                
    def check_amplitude(self, gate, mvrange):
        """ Calculates the lowest allowable AWG peak-to-peak amplitude based on the
        ranges to be applied to the gates. If the AWG amplitude is too low, it gives
        a warning and increases the amplitude.
        
        Args:
            gate (str): name of the gate to check
            mvrange (float): voltage range, in mV, that the gate needs to reach
        """
        min_amp = mvrange / self.hardware.parameters['awg_to_%s' % gate].get()
        if min_amp > 4:
            raise(Exception('Sweep range of gate %s is larger than maximum allowed by the AWG' % gate))
        if self.ch_amp < min_amp:
            min_amp = np.ceil(min_amp * 10) / 10
            self.set_amplitude(min_amp)
            warnings.warn('AWG amplitude too low for this range, setting to %.1f' % min_amp)

#%%


def plot_wave_raw(wave_raw, samplerate=None, station=None):
    ''' Plot the raw wave 

    Arguments:
        wave_raw (array): raw data which represents the waveform

    Returns:
        plot (QtPlot): the plot showing the data
    '''
    if samplerate is None:
        if station is None:
            raise Exception('There is no station')
        samplerate = 1 / station.awg.getattr('AWG_clock')
    else:
        samplerate = samplerate
    horz_var = np.arange(0, len(wave_raw) * samplerate, samplerate)
    x = DataArray(name='time(s)', label='time (s)',
                  preset_data=horz_var, is_setpoint=True)
    y = DataArray(
        label='sweep value (mV)', preset_data=wave_raw, set_arrays=(x,))
    plot = QtPlot(x, y)

    return plot


def sweep_2D_process(data, waveform, diff_dir=None):
    ''' Process data from sweep_2D 

    Arguments:
        data (list): the raw measured data
        waveform (dict): The waveforms that was sent with the AWG.

    Returns:
        data_processed (list): the processed data
    '''
    width_horz = waveform['width_horz']
    width_vert = waveform['width_vert']
    resolution = waveform['resolution']

    # split up the fpga data in chunks of horizontal sweeps
    chunks_ch1 = [data[x:x + resolution[0]] for x in range(0, len(data), resolution[0])]
    chunks_ch1 = [chunks_ch1[i][1:int(width_horz * len(chunks_ch1[i]))] for i in range(0, len(chunks_ch1))]
    data_processed = chunks_ch1[:int(width_vert * len(chunks_ch1))]

    if diff_dir is not None:
        data_processed = qtt.utilities.tools.diffImageSmooth(data_processed, dy=diff_dir, sigma=1)

    return data_processed
