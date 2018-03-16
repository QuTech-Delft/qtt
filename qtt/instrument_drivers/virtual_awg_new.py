import logging
import numpy as np

from enum import Enum
from qctoolkit.pulses import SequencePT, TablePT
from qctoolkit.pulses.sequencing import Sequencer
from qctoolkit.pulses.plotting import plot, render
from qcodes.instrument.base import InstrumentBase


# -----------------------------------------------------------------------------
# qc-toolkit template pulses... separate file?


def pulsewave_template(name: str='pulse'):
    return TablePT({name: [(0, 'amplitude'), ('width', 0), ('holdtime', 0)]})


def sawtooth_template(name: str='sawtooth'):
    return TablePT({name: [(0, 0), ('period/4', 'amplitude', 'linear'),
                           ('period*3/4', '-amplitude', 'linear'),
                           ('period', 0, 'linear')]})


def wait_template(name: str='wait'):
    return TablePT({name: [(0, 0), ('holdtime', 0)]})


def to_array(sequence: SequencePT, sample_rate: int):
    sequencer = Sequencer()
    sequencer.push(sequence)
    build = sequencer.build()
    if not sequencer.has_finished():
        raise ValueError
    return render(build, sample_rate)[1]

# -----------------------------------------------------------------------------
# qc-toolkit stuff... separate file?

class DataType(Enum):
    RAW_DATA = 0
    QC_TOOLKIT = 1


def make_sawtooth(vpp, period, width=0, reps=1):
    values = {'period': period, 'amplitude': vpp/2}
    data = (sawtooth_template(), values)
    return {'TYPE': DataType.QC_TOOLKIT, 'WAVE': SequencePT(*(data,)*reps)}


def make_pulses(voltages, waittimes, filter_cutoff=None, mvrange=None):
    sequence = []
    return sequence


def sequence_to_waveform(sequence: SequencePT, sample_rate: int):
    '''generates waveform from sequence.
    '''
    sequencer = Sequencer()
    sequencer.push(sequence)
    build = sequencer.build()
    if not sequencer.has_finished():
        raise ValueError
    voltages = render(build, sample_rate)[1]
    return voltages[next(iter(voltages))]  # ugly!!!


def plot_waveform(waveform, sample_rate: int):
    '''plots the waveform array.'''
    from matplotlib import pyplot as plt
    sample_count = len(waveform)
    total_time = sample_count / sample_rate
    times = np.linspace(0, total_time, num=sample_count)
    plt.plot(times, waveform)


def plot_sequence(sequence, sample_rate):
    '''plots the qc-toolkit sequence.'''
    plot(sequence['WAVE'], sample_rate)


'''
s = make_sawtooth('sample_01', 1, 1, 2)
plot_sequence(s, sample_rate=1e4)
d = sequence_to_waveform(s, sample_rate=1e4)
'''

# -----------------------------------------------------------------------------


class VirtualAwg(InstrumentBase):

    def __init__(self, instruments, parameters, name='virtual_awg', **kwargs):
        ''' Creates a new VirtualAWG.

        Arguments:
            name (str): the name of the virtual awg.
            instruments (list): a list with qcodes instruments (implemented
                awgs and digitizers will be correctly picked out).
            parameters (Parameters): parameter object with properties (e.g.
                awg_map, clock_speed, marker_delay, etc.).
        '''
        super().__init__(name, **kwargs)
        self.awgs, self.digitizer = self.__set_hardware(instruments)
        self.parameters = parameters
        self.__check_parameters()
        self.__preset_awgs()

    def __set_hardware(self, instruments):
        ''' Adds the digitizers and awg's to the virual awg. For each device is
            used via a derived VirtualAwgBase or VirtualDigitizerBase class.
            This is done because we would like to use the same functionality
            for different awg's and digizers with different command structures.

        Arguments:
            instruments (list): a list with qcodes instruments (implemented
                awgs and digitizers. order must match!!!).

        Returns:
            (awgs, digitizer) (list, object): the VirtualAwgBase objects with
            the awgs as a list and the VirtualDigitizerBase with the digitzer.

        '''
        awgs, digitizer = [], None
        for device in instruments:
            if type(device).__name__ == 'Tektronix_AWG5014':
                awgs.append(TektronixVirtualAwg(device))
            elif type(device).__name__ == 'M3201A':
               awgs.append(KeysightVirtualAwg(device))
            elif type(device).__name__ == 'M4i':
                digitizer = M4iVirtualDigitizer(device)
            else:
                raise VirtualAwgError('Unusable instrument added!')
        digitizer = M4iVirtualDigitizer(None) #  TODO For testing...
        return awgs, digitizer

    def __check_parameters(self):
        ''' Checks whether the provided parameter object has the minimal
            required values for using the virtual AWG. TODO !!!
        '''
        pass

    def __preset_awgs(self):
        ''' Sets the awg's in continues or sequence mode, depending on the
            amount and order of the presented awg's. The clock speed,
            channel amplitudes will be set and all waveforms removed,
            if one or two awg's are connected.
        '''
        awg_count = len(self.awgs)
        if awg_count == 0:
            logging.warning("No physical awg's connected!")
            return
        if awg_count == 1:
            self.awgs[0].set_as_master()
            logging.info("One physical awg's connected!")
        elif awg_count == 2:
            (awg_nr, _, _) = self.parameters.awg_map['awg_mk']
            self.awgs[awg_nr].set_as_master()
            self.awgs[awg_nr + 1 % 2].set_as_slave()
            logging.info("Two physical awg's connected!")
        else:
            class_name = self.__class__.__name__
            message = "Configuration not supported by {0}!".format(class_name)
            logging.error(message)
            raise VirtualAwgError(message)
        for awg in self.awgs:
            awg.set_amplitudes()
            awg.set_sampling_frequency()
            awg.delete_all_waveforms()

    def are_awg_gates(self, gate_s):
        ''' Returns true if the gate or all given gates can be controlled
            by the awg(s).

        Arguments:
            gate_s (list or str): the name or names of the gate(s).

        Example:
        -------
        >>> result = <VirtualAwg>.are_awg_gates('X1')
        >>> result = <VirtualAwg>.are_awg_gates(['X1', 'P2', 'P6'])
        '''
        if gate_s is None:
            return False
        if self.parameters.awg_map is None:
            return False
        if isinstance(gate_s, list):
            return np.all([self.are_awg_gates(g) for g in gate_s])
        return True if gate_s in self.parameters.awg_map else False

    def stop_awgs(self):
        ''' Stops all awg(s) and turns off all channels.'''
        [awg.stop() for awg in self.awgs]
        logging.info("All awg's stopped...")

    def reset_awgs(self):
        ''' Resets all awg(s) and turns of all channels.'''
        [awg.reset() for awg in self.awgs]
        logging.info("All awg's are reseted...")

    def __get_raw_data(self, waveform, sampling_rate):
        ''' This function returns the raw array data given the waveform.
            A waveform can hold different types of data dependend on the
            used pulse library. Currently only raw array data and QC-toolkit
            can be used.

        Arguments:
            waveform (dict): a waveform is a dictionary with "type" value
            given the used pulse library. The "wave" value should contain
            the actual wave-object.
            sampling_rate: a sample rate of the awg in samples per sec.

        Returns:
            A numpy.ndarrays with the corresponding sampled voltages.
        '''
        data_type = waveform['TYPE']
        if data_type == DataType.RAW_DATA:
            return waveform['WAVE']
        if data_type == DataType.QC_TOOLKIT:
            return sequence_to_waveform(waveform['WAVE'], sampling_rate/10**9)

    def sweep_init(self, waveforms, do_upload=True):
        '''Sends the waveform(s) to gate(s).

        Arguments:
            waveforms (dict): the waveforms with the gates as keys.
            do_upload (bool): indicated whether the waveforms are directly
                              uploaded.
            sample_frequency (float): sets the pre-trigger period.

        Returns:
            sweep_info (dict): a dictionary with awgs channels as keys and
                waveforms and marker data as values.

        Example:
        -------
        >>> sweep_info = <VirtualAwg>.sweep_init(waveforms)
        '''
        self.digitizer.set_marker(self.parameters)
        # make sure all waveforms are of equal length.

        sweep_info = dict()
        for gate, wave in waveforms.items():

            # awg wave
            awg_nr, awg_ch = self.parameters.awg_map[gate]
            sample_freq = self.awgs[awg_nr].get_sampling_frequency()
            raw_wave = self.__get_raw_data(wave, sample_freq)
            raw_wave_length = len(raw_wave)
            sweep_info[(awg_nr, awg_ch)] = {'name': 'waveform_' + gate, 'waveform': raw_wave,
                                            'marker1': np.zeros(raw_wave_length),
                                            'marker2': np.zeros(raw_wave_length)}

            # awg marker
            mrk_awg, mkr_ch, mrk_nr = self.digitizer.marker_channel
            marker_delay = self.digitizer.marker_delay()
            if (mrk_awg, mkr_ch) in waveforms.values():
                sweep_info[(mrk_awg, mkr_ch)]['delay'] = marker_delay
            else:
                sweep_info[(mrk_awg, mkr_ch)] = {'name': self.digitizer.marker_name,
                                                 'waveform': np.zeros(raw_wave_length),
                                                 'marker1': np.zeros(raw_wave_length),
                                                 'marker2': np.zeros(raw_wave_length),
                                                 'delay': marker_delay}
            marker_points = np.zeros(raw_wave_length)
            sample_freq = self.awgs[awg_nr].get_sampling_frequency()
            idx_start = int(marker_delay * sample_freq)
            marker_points[idx_start:idx_start+raw_wave_length//20] = 1.0  # create marker_waveform
            sweep_info[(mrk_awg, mkr_ch)]['marker%d' % mrk_nr] = marker_points
            marker_name = 'ch{0}_m{1}'.format(mrk_awg, mrk_nr)
            self.awgs[mrk_awg].set_marker(marker_name, self.digitizer.marker_low,
                                          self.digitizer.marker_high)

            # seq awg marker
            if len(self.awgs) == 2:
                name = 'awg_mk'
                sawg_nr, sawg_ch, sawg_mrk = self.parameters.awg_map[name]
                if (sawg_nr, sawg_ch) not in sweep_info:
                    sweep_info[sawg_ch] = {'name': name,
                                           'waveform': np.zeros(raw_wave_length),
                                           'marker1': np.zeros(raw_wave_length),
                                           'marker2': np.zeros(raw_wave_length)}
                awg_marker_data = np.zeros(raw_wave_length)
                awg_marker_data[0: raw_wave_length//20] = 1.0
                delta = int(self.awgs[awg_nr].delay - self.awgs[awg_nr].clock_speed)
                awg_marker_data = np.roll(awg_marker_data, raw_wave_length - delta)
                marker_name = 'ch{0}_m{1}'.format(sawg_nr, sawg_ch)
                self.awgs[sawg_nr].set_marker(marker_name, self.digitizer.marker_low,
                                              self.digitizer.marker_high)

        if do_upload:
            try:
                [awg.delete_all_waveforms() for awg in self.awgs]
                for (awg_nr, awg_channel), info in sweep_info.items():
                    self.awgs[awg_nr].send_waveform(info['name'], info['waveform'],
                                                    info['marker1'], info['marker2'])
            except Exception as ex:
                logging.error('(%s, %s, %s, %s) = %s', info['waveform'].shape,
                             info['marker1'].shape, info['marker2'].shape, info['name'], ex)
                raise

        return sweep_info

    def sweep_run(self, sweep_info):
        ''' Activate AWG(s) and channel(s) for the sweep(s).

        Arguments:
            sweep_info (dict): a dictionary with awgs channels as keys and
                waveforms and marker data as values.
        '''
        for (awg_nr, awg_channel), info in sweep_info.items():
            awg = self.awgs[awg_nr]
            if len(self.awgs) == 2 and self.awgs[1] == self.awgs[awg_channel]:
                awg.get.set_sqel_waveform(info['name'], awg_channel, 1)
                awg.get.set_sqel_loopcnt_to_inf(1)
                awg.get.set_sqel_event_jump_target_index(awg_channel, 1)
                awg.get.set_sqel_event_jump_type(1, 'IND')
            else:
                awg.get.set('ch{}_waveform'.format(awg_channel), info['name'])

        for (awg_nr, awg_channel) in sweep_info.keys():
            self.awgs[awg_nr].get.set('ch{}_state'.format(awg_channel), 1)

        for (awg_nr, awg_channel) in sweep_info.keys():
            self.awgs[awg_nr].get.run()
        return

    def sweep_process(self, data, waveform, averages=1, direction='forwards', start_offset=1):
        '''Process the returned data using shape of the sawtooth send with the AWG.'''
        pass

    def sweep_gates(self, gate_s, sweeprange, period, width=.95, do_upload=True):
        ''' Send a sawtooth signal with the AWG to a linear combination of 
        gates to sweep. Also send a marker to the measurement instrument.

        Arguments:
            gate_s (str or dict): the gates to sweep and the coefficients as values
            sweeprange (float): the range of voltages to sweep over
            period (float): the period of the triangular signal

        Returns:
            waveform (dict): The waveform being send with the AWG.
            sweep_info (dict): the keys are tuples of the awgs and channels to activate
        '''

        if not self.are_awg_gates(gate_s):
            raise VirtualAwgError('Unusable gate selected!')

        if type(gate_s) == str:
            gate_s = [gate_s]

        waveforms = dict()
        for gate in gate_s:
            #self.check_amplitude(g, gate_comb[g] * sweeprange)
            saw_tooth = make_sawtooth(sweeprange, period)
            #awg_to_plunger = self.hardware.parameters['awg_to_%s' % gate].get()
            #wave = wave_raw * gate_s[gate] / awg_to_plunger
            waveforms[gate] = saw_tooth

        sweep_info = self.sweep_init(waveforms, do_upload)
        self.sweep_run(sweep_info)
        waveforms['width'] = width
        waveforms['sweeprange'] = sweeprange
        waveforms['samplerate'] = 1 / self.awgs[0].get_sampling_frequency()
        waveforms['period'] = period
        #for channels in sweep_info:
        #    if 'delay' in sweep_info[channels]:
        #        waveforms['markerdelay'] = sweep_info[channels]['delay']
        return waveforms, sweep_info

    # should be replaced by adding waveforms in the qc_toolkit framework?
    #@qtt.tools.deprecated
    #def sweepandpulse_gate(self, sweepdata, pulsedata, wave_name=None, delete=True):
    #    pass

    def sweep_gates_2D(self, gates_horz, gates_vert, sweepranges, resolution, width=.95, do_upload=True):
        pass

    def sweep_2D_process(self, data, waveform, diff_dir=None):
        pass

    def pulse_gates(self, gate_voltages, waittimes, filtercutoff=None, delete=True):
        pass

    # FIXME: keep?
    def set_amplitude(self, amplitude):
        pass

    def check_amplitude(self, gate, mvrange):
        pass

    def __check_frequency_waveform(self, period, width):
        pass

# -----------------------------------------------------------------------------


class RunMode(Enum):
    SEQ = 'SEQ'
    CONT = 'CONT'


class VirtualAwgBase():

    channels = 4
    markers = 2

    def set_master(self):
        pass
    
    def set_slave(self):
        pass

    def set_mode(self, value):
        raise NotImplementedError

    def set_sequence_mode(self, value: RunMode):
        raise NotImplementedError

    def delete_all_waveforms(self):
        raise NotImplementedError

    def send_waveform(self, name, waveform, marker1=None, marker2=None):
        raise NotImplementedError

    def set_amplitudes(self, value):
        raise NotImplementedError

    def set_clock_speed(self, value):
        raise NotImplementedError

    def prepare_run(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class DigitizerBase():

    def set_marker(self, properties):
        raise NotImplementedError

# -----------------------------------------------------------------------------


class TektronixVirtualAwg(VirtualAwgBase):

    def __init__(self, awg):
        self.__awg = awg

    @property
    def get(self):
        return self.__awg

    def set_as_master(self):
        self.__awg.set('run_mode', 'CONT')

    def set_as_slave(self):
        self.__awg.set('run_mode', 'SEQ')
        self.__awg.sequence_length.set(1)
        self.__awg.set_sqel_trigger_wait(1, 0)

    def set_sampling_frequency(self, value=1e9):
        self.__awg.set('clock_freq', value)

    def get_sampling_frequency(self):
        return self.__awg.get('clock_freq')

    def set_amplitudes(self, value=4.0):
        map(lambda i: self.__awg.set('ch{}_amp'.format(i), value), range(1, 5))

    def set_marker(self, name, low, high):
        self.__awg.parameters[name + '_low'] = low
        self.__awg.parameters[name + '_high'] = high

    def stop(self):
        self.__awg.stop()

    def reset(self):
        self.__awg.reset()

    def delete_all_waveforms(self):
        self.__awg.delete_all_waveforms_from_list()

    def send_waveform(self, name, waveform, marker1, marker2):
        self.__awg.send_waveform_to_list(waveform, marker1, marker2, name)


# -----------------------------------------------------------------------------


class KeysightVirtualAwg(VirtualAwgBase):
    pass

# -----------------------------------------------------------------------------


class M4iVirtualDigitizer(DigitizerBase):

    def __init__(self, digitizer):
        self.__digitizer = digitizer

    def set_marker(self, properties):
        names = [key for key in properties.awg_map.keys() if key.startswith('dig_')]
        if len(names) == 1:
            self.marker_name = names[0]
            self.marker_delay = properties.marker_delay
            self.marker_channel = properties.awg_map[self.marker_name]
            self.marker_low = 0.0
            self.marker_high = 2.6
            return
        raise KeyError('No digital marker in parameters!')

# -----------------------------------------------------------------------------


class VirtualAwgError(Exception):
    '''Exception for a specific error related to the virual AWG.'''