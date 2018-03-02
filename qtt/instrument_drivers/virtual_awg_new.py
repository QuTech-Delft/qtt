import logging
import numpy as np

from enum import Enum
from qctoolkit.pulses import SequencePT, TablePT
from qctoolkit.pulses.sequencing import Sequencer
from qctoolkit.pulses.plotting import plot, render

from qcodes.instrument.base import InstrumentBase
from qcodes.instrument_drivers.Spectrum import M4i
from qcodes.instrument_drivers.Keysight import M3201A
from qcodes.instrument_drivers.tektronix import AWG5014

logger = logging.getLogger(__name__)

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


def make_sawtooth(vpp, period, width, reps=1):
    values = {'period': period, 'amplitude': vpp/2}
    data = (sawtooth_template(), values)
    return SequencePT(*(data,)*reps)  # to dict with additional data.


def make_pulses(voltages, waittimes, filter_cutoff=None, mvrange=None):
    sequence = []
    return sequence


def sequence_to_waveform(sequence: SequencePT, sample_rate: int):
    '''generates waveform from sequence'''
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
    plot(sequence, sample_rate)


'''
s = make_sawtooth(1, 1, 1, 2)
plot(s, sample_rate=1e4)
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
        self.awgs, self.digitizer = VirtualAwg.__set_hardware(instruments)
        super().__init__(name, **kwargs)
        self.parameters = parameters
        self.__check_parameters()
        self.__preset_awgs()

    @staticmethod
    def __set_hardware(instruments):
        awgs, digitizer = [], None
        for device in instruments:
            if isinstance(device, AWG5014):
                awgs.append(TektronixVirtualAwg(device))
            elif isinstance(device, M3201A):
                awgs.append(KeysightVirtualAwg(device))
            elif isinstance(device, M4i):
                if digitizer:
                    raise VirtualAwgError((device, digitizer),
                                          'Multiple digitizers not supported!')
                digitizer = M4iVirtualDigitizer(device)
            else:
                continue
        return awgs, digitizer

    def __check_parameters(self):
        pass  # TODO !!!

    def __preset_awgs(self):
        awg_count = len(self.awgs)
        if awg_count == 0:
            logger.warning("No physical awg's connected!")
            return
        if awg_count == 1:
            self.awgs[0].set_mode(RunMode.CONT)
            logger.verbose("One physical awg's connected!")
        elif awg_count == 2:
            (awg_nr, _, _) = self.parameters.awg_map['awg_mk']
            self.awgs[awg_nr].set_mode(RunMode.CONT)
            self.awgs[awg_nr + 1 % 2].seq_awg.set_mode(RunMode.SEQ)  # Why +1%2???
            logger.verbose("Two physical awg's connected!")
        else:
            class_name = self.__class__.__name__
            message = "Configuration not supported by {0}!".format(class_name)
            logger.error(message)
            raise VirtualAwgError(message)
        for awg in self.awgs:
            awg.set_clock_speed(self.parameters.clock_speed)
            awg.set_amplitudes(self.parameters.channel_amplitude)
            awg.delete_all_waveforms()
        return

    def are_awg_gates(self, gate_s):
        ''' Return true if the gate or all given gates can be controlled
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
        if isinstance(gate_s, dict):
            return np.all([self.is_awg_gate(g) for g in gate_s])
        return True if gate_s in self.parameters.awg_map else False

    def reset_awgs(self):
        ''' Resets all awg(s) and turns of all channels.
        '''
        map(lambda awg: awg.reset(), self.awgs)
        logger.verbose("All awg's are reseted...")

    def stop_awgs(self):
        ''' Stops all awg(s) and turns of all channels.
        '''
        map(lambda awg: awg.get.stop(), self.awgs)
        logger.verbose("All awg's stopped...")

    @staticmethod
    def __get_data(waveform):
        data_type = waveform['type']
        if data_type == DataType.RAW_DATA:
            return waveform['wave']
        if data_type == DataType.QC_TOOLKIT:
            sample_rate = 1  # TODO GS/s !!!
            return sequence_to_waveform(waveform['wave'], sample_rate)

    def sweep_init(self, waveforms, do_upload=True, sample_frequency=None):
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
        data_count = len(waveforms[waveforms.keys()[0]]['wave'])  # ugly!!!
        #  make sure all waveforms are of equal length.
        self.digitizer.set_marker(self.properties, sample_frequency)
        sweep_info = dict()
        # wave data
        for gate, wave in waveforms:
            awg_channel = self.properties.awg_map[gate]
            sweep_info[awg_channel] = {'name': wave['name'],
                                       'waveform': VirtualAwg.__get_data(wave),
                                       'marker1': np.zeros(data_count),
                                       'marker2': np.zeros(data_count)}
        # marker points
        awg_marker = self.digitizer.marker_channel[:2]
        marker_delay = self.digitizer.marker_delay
        marker_points = np.zeros(data_count)
        idx_start = int(marker_delay * self.parameters.clock_speed)
        marker_points[idx_start:idx_start+data_count//20] = 1.0
        if awg_marker in waveforms.values():
            sweep_info[awg_marker]['delay'] = marker_delay
        else:
            sweep_info[awg_marker] = {'name': self.digitizer.marker_name,
                                      'waveform': np.zeros(data_count),
                                      'marker1': np.zeros(data_count),
                                      'marker2': np.zeros(data_count),
                                      'delay': marker_delay}
        marker_nr = self.digitizer.marker_channel[2]
        sweep_info[awg_marker][marker_nr] = marker_points
        m_name = 'ch{0}_m{1}'.format(awg_marker[0], awg_marker[1])
        self.awgs[marker_nr].set(m_name + '_low', self.digitizer.marker_low)
        self.awgs[marker_nr].set(m_name + '_high', self.digitizer.marker_high)

        # awg marker
        if self.seq_awg:
            name = 'awg_mk'
            awg_info = self.parameters.awg_map[name]
            channel = awg_info[:2]
            if channel not in sweep_info:
                awg_nr = awg_info[0]
                sweep_info[channel] = {'name': name,
                                       'waveform': np.zeros(data_count),
                                       'marker1': np.zeros(data_count),
                                       'marker2': np.zeros(data_count)}
            awg_marker_data = np.zeros(data_count)
            awg_marker_data[0: data_count//20] = 1
            delta = int(self.awgs[awg_nr].delay -
                        self.awgs[awg_nr].clock_speed)
            awg_marker_data = np.roll(awg_marker_data, data_count - delta)
            a_name = 'ch{0}_m{1}'.format(channel[0], channel[1])
            self.awgs[awg_nr].set(a_name + '_low', self.digitizer.marker_low)
            self.awgs[awg_nr].set(a_name + '_high', self.digitizer.marker_high)

        if do_upload:
            try:
                map(lambda awg: awg.delete_all_waveforms(), self.awgs)
                for(awg_nr, awg_channel), info in sweep_info.items():
                    self.awgs[awg_nr].send_waveform_to_list(info['waveform'],
                                                            info['marker1'],
                                                            info['marker2'],
                                                            info['name'])
            except Exception as ex:
                logger.error('({0}, {1}, {2}) = {3}'.format(
                    info['waveform'].shape, info['marker1'].shape,
                    info['marker2'].shape, ex))
                raise VirtualAwgError(ex)

        return sweep_info

    def sweep_run(self, sweep_info):
        ''' Activate AWG(s) and channel(s) for the sweep(s).

        Arguments:
            sweep_info (dict): a dictionary with awgs channels as keys and
                waveforms and marker data as values.
        '''
        for (awg_nr, awg_channel), info in sweep_info.items():
            awg = self.awgs[awg_nr]
            if self.seq_awg and self.seq_awg == self.awgs[awg_channel]:
                awg.set_sqel_waveform(info['name'], awg_channel, 1)
                awg.set_sqel_loopcnt_to_inf(1)
                awg.set_sqel_event_jump_target_index(awg_channel, 1)
                awg.set_sqel_event_jump_type(1, 'IND')
            else:
                awg.set('ch{}_waveform'.format(awg_channel), info['name'])

        for (awg_nr, awg_channel), info in sweep_info:
            self.awgs[awg_nr].set('ch{}_state'.format(awg_channel), 1)

        for awg_nr in sweep_info.keys:
            self.awgs[awg_nr].get.run()
        return

    def sweep_process(self, data, waveform, Naverage=1, direction='forwards', start_offset=1):
        '''Process the returned data using shape of the sawtooth send with the AWG.'''
        pass

    # FIXME: make use if sweep_gate_virt
    def sweep_gate(self, gate, sweeprange, period, width=.95, wave_name=None, delete=True):
        pass

    def sweep_gate_virt(self, gate_comb, sweeprange, period, width=.95, delete=True):
        pass

    # should be replaced by adding waveforms in the qc_toolkit framework?
    #@qtt.tools.deprecated
    #def sweepandpulse_gate(self, sweepdata, pulsedata, wave_name=None, delete=True):
    #    pass

    # FIXME: make use of sweep_2D_virt
    # JP: I think FPGA exceptions should not be handled by awg
    #        if resolution[0] * resolution[1] > self.maxdatapts:
    #            raise Exception('resolution is set higher than FPGA memory allows')
    def sweep_2D(self, samp_freq, sweepgates, sweepranges, resolution, width=.95, comp=None, delete=True):
        pass

    def sweep_2D_virt(self, samp_freq, gates_horz, gates_vert, sweepranges, resolution, width=.95, delete=True):
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


class DataType(Enum):
    RAW_DATA = 0
    QC_TOOLKIT = 1


class VirtualAwgBase():

    channels = 4
    markers = 2

    def set_continues_mode(self):
        raise NotImplementedError

    def set_mode(self, value):
        raise NotImplementedError

    def set_sequence_mode(self, value: RunMode):
        raise NotImplementedError

    def delete_all_waveforms(self):
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

    def set_mode(self, value: RunMode):
        self.__awg.set('run_mode', value.name)
        if value == RunMode.SEQ:
            self.__awg.sequence_length.set(1)
            self.__awg.set_sqel_trigger_wait(1, 0)
        return

    def set_clock_speed(self, value=1e8):
        self.__awg.set('clock_freq', value)

    def set_amplitudes(self, value=4.0):
        map(lambda i: self.__awg.set('ch{}_amp'.format(i), value),
            range(1, 5))

    def stop():
        self.__awg.stop()
        map(lambda i: self.__awg.set('ch{}_state'.format(i)), range(1, 5))


class KeysightVirtualAwg(VirtualAwgBase):
    pass

# -----------------------------------------------------------------------------


class M4iVirtualDigitizer(DigitizerBase):

    def __init__(self, digitizer):
        self.__digitizer = digitizer

    def set_marker(self, properties):
        names = [key.startswith('dig_') for key in properties.awg_amp]
        if len(names) == 1:
            self.marker_name = names[0]
            self.marker_delay = properties.marker_delay
            self.marker_channel = properties.awg_map[self.marker_name]
            self.marker_low = 0.0
            self.marker_high = 2.6
            return
        raise(KeyError)

# -----------------------------------------------------------------------------


class VirtualAwgError(Exception):
    '''Exception for a specific error related to the virual AWG.'''