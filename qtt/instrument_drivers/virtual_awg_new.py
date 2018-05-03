import logging
import itertools
import numpy as np
from enum import Enum
from functools import reduce
from matplotlib import pyplot as plt

from qcodes.instrument.base import InstrumentBase
from qctoolkit.pulses import SequencePT, TablePT
from qctoolkit.pulses.sequencing import Sequencer
from qctoolkit.pulses.plotting import plot, render, PlottingNotPossibleException

# %%-----------------------------------------------------------------------------------------------


class DataType(Enum):
    RAW_DATA = 0
    QC_TOOLKIT = 1


def qc_toolkit_template_to_array(template, sampling_rate, parameters=None):
    """ Renders the QC toolkit template as voltages array.

    Arguments:
        template (*PT): A QC Toolkit template of type; pulsePT,
                        functionPT, pointPT or sequencePT, or other.
        samples_per_ns (float): The number of samples per nanosecond.

    Returns:
        voltages (np.array): The array with voltages generated from
                             the template.
    """
    channels = template.defined_channels
    if parameters is None:
        parameters = dict()
    sequencer = Sequencer()
    sequencer.push(template, parameters,
                   channel_mapping={ch: ch for ch in channels},
                   window_mapping={w: w for w in template.measurement_names})
    sequence = sequencer.build()
    if not sequencer.has_finished():
        raise PlottingNotPossibleException(template)
    (_, voltages) = render(sequence, sampling_rate/1e9)
    return voltages[next(iter(voltages))]


def get_raw_data(waveform, sampling_rate):
    """ This function returns the raw array data given the waveform.
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
    """
    data_type = waveform['TYPE']
    if data_type == DataType.RAW_DATA:
        return waveform['WAVE']
    if data_type == DataType.QC_TOOLKIT:
        return qc_toolkit_template_to_array(waveform['WAVE'], sampling_rate)


def plot_waveform_array(array, sampling_rate):
    """ Plots the waveform array."""
    sample_count = len(array)
    total_time = (sample_count - 1)/sampling_rate*1e9
    times = np.linspace(0, total_time, num=sample_count, dtype=float)
    plt.step(times, array, where='post')
    plt.show()


def plot_sequence(sequence, sampling_rate):
    """ Plots the qc-toolkit sequence."""
    plot(sequence['WAVE'], sample_rate=sampling_rate/1e9)


def pulsewave_template(name='pulse'):
    return TablePT({name: [(0, 'amplitude'), ('width', 0), ('holdtime', 0)]})


def sawtooth_template(name='sawtooth'):
    return TablePT({name: [(0, 0), ('period/4', 'amplitude', 'linear'),
                           ('period*3/4', '-amplitude', 'linear'),
                           ('period', 0, 'linear')]})


def wait_template(name: str='wait'):
    return TablePT({name: [(0, 0), ('holdtime', 0)]})


def make_sawtooth(vpp, period, repetitions=1, name='sawtooth'):
    seq_data = (sawtooth_template(), {'period': period*1e9, 'amplitude': vpp/2})
    return {'NAME': name, 'WAVE': SequencePT(*((seq_data,)*repetitions)),
            'TYPE': DataType.QC_TOOLKIT}

def test_make_sawtooth_HasCorrectProperties():
    sampling_rate = 1e9
    sawtooth_sequence = make_sawtooth(1.5, period=1e-7)
    plot_sequence(sawtooth_sequence, sampling_rate)

    array = get_raw_data(sawtooth_sequence, sampling_rate)
    plot_waveform_array(array, sampling_rate)

# %%-----------------------------------------------------------------------------------------------


class VirtualAwg(InstrumentBase):

    def __init__(self, instruments, parameters, name='virtual_awg', **kwargs):
        """ Creates a new VirtualAWG.

        Arguments:
            name (str): the name of the virtual awg.
            instruments (list): a list with qcodes instruments (implemented
                awgs and digitizers will be correctly picked out).
            parameters (Parameters): parameter object with properties (e.g.
                awg_map, clock_speed, marker_delay, etc.).
        """
        super().__init__(name, **kwargs)
        self.awgs = self.__set_hardware(instruments)
        self.parameters = parameters
        self.__check_parameters()
        self.__preset_awgs()

    def __set_hardware(self, instruments):
        """ Adds the digitizers and awg's to the virual awg. For each device is
            used via a derived VirtualAwgBase or VirtualDigitizerBase class.
            This is done because we would like to use the same functionality
            for different awg's and digizers with different command structures.

        Arguments:
            instruments (list): a list with qcodes instruments (implemented
                awgs and digitizers. order must match!!!).

        Returns:
            (awgs, digitizer) (list, object): the VirtualAwgBase objects with
            the awgs as a list and the VirtualDigitizerBase with the digitzer.

        """
        awgs = []
        for device in instruments:
            if type(device).__name__ == 'Tektronix_AWG5014':
                awgs.append(TektronixVirtualAwg(device))
            elif type(device).__name__ == 'M3201A':
                awgs.append(KeysightVirtualAwg(device))
            else:
                raise VirtualAwgError('Unusable instrument added!')
        return awgs

    def __check_parameters(self):
        """ Checks whether the provided parameter object has the minimal
            required values for using the virtual AWG. Needed ???
        """
        pass

    def __preset_awgs(self):
        ''' Sets the awg's in continues or sequence mode, depending on the
            amount and order of the presented awg's. The clock speed,
            channel amplitudes will be set and all waveforms removed,
            if one or two awg's are connected.
        '''
        self.awg_count = len(self.awgs)
        if self.awg_count == 0:
            logging.warning("No physical awg's connected!")
            return
        if self.awg_count == 1:
            self.awgs[0].set_sequence_mode()
            logging.info("One physical awg's connected!")
        elif self.awg_count == 2:
            (awg_nr, _, _) = self.parameters.awg_map['awg_mk']
            self.awgs[awg_nr].set_sequence_mode()
            self.awgs[awg_nr + 1 % 2].set_sequence_mode()
            logging.info("Two physical awg's connected!")
        else:
            class_name = self.__class__.__name__
            message = "Configuration not supported by {0}!".format(class_name)
            logging.error(message)
            raise VirtualAwgError(message)
        [awg.set_awg_properties(self.parameters) for awg in self.awgs]

    def are_awg_gates(self, gate_s):
        """ Returns true if the gate or all given gates can be controlled
            by the awg(s).

        Arguments:
            gate_s (list or str): the name or names of the gate(s).

        Example:
        -------
        >>> result = <VirtualAwg>.are_awg_gates('X1')
        >>> result = <VirtualAwg>.are_awg_gates(['X1', 'P2', 'P6'])
        """
        if gate_s is None:
            return False
        if self.parameters.awg_map is None:
            return False
        if isinstance(gate_s, list):
            return np.all([self.are_awg_gates(g) for g in gate_s])
        return True if gate_s in self.parameters.awg_map else False

    def run_awgs(self):
        """ Turns on the AWG channels selected using the gates."""
        [awg.run() for awg in self.awgs]

    def enable_channels_outputs(self, awg_nr=None, channels=None):
        if awg_nr:
            self.awgs[awg_nr].set_state_channels(True, channels)
        else:
            [awg.set_state_channels(True, channels) for awg in self.awgs]
        logging.info("Enabled AWG outputs.")

    def disable_channels_outputs(self, awg_nr=None, channels=None):
        if awg_nr:
            self.awgs[awg_nr].set_state_channels(False, channels)
        else:
            [awg.set_state_channels(False, channels) for awg in self.awgs]
        logging.info("Disabled AWG outputs.")

    def stop_awgs(self):
        """ Stops all awg(s) and turns off all channels."""
        [awg.stop() for awg in self.awgs]
        logging.info("All awg's stopped...")

    def reset_awgs(self):
        """ Resets all awg(s) and turns of all channels."""
        [awg.reset() for awg in self.awgs]
        logging.info("All awg's are reseted...")

    def sequence_gates(self, gates, waveforms, repetitions, gotos):
        # TODO check lengths...

        all_outputs = [dict() for awg in self.awg_count]
        for gate in gates:
            (awg_nr, channel_nr, *marker_nr) = self.parameters.awg_map[gate]
            all_outputs[awg_nr].keys()



    def reorder_sequence(self, names, repetitions, gotos):
        pass


    def sweep_init(self, gates, waveforms, do_upload=True, period=None, delete=None, samp_freq=None):
        """ Sends the waveform(s) to gate(s) and markers(s).

        Arguments:
            waveforms (dict): the waveforms with the gates/markers as keys. Note markers must have
                              waveforms with only 0 and 1's.
            do_upload (bool): indicates whether the waveforms are uploaded.
            period, delete, sample_freq: depricated arguments. To be removed later!

        Example:
        -------
        >>> gates = ['X1', 'P1', 'mk1']
        >>> waveforms = [[seq1, seq2],[seq3, seq4], [seq2, seq1]]
        >>> <VirtualAwg>.sweep_init(gates, waveforms)
        """
        if period or delete or samp_freq:
            logging.error('Arguments: period, delete, samp_rate are depricated!')
            raise VirtualAwgError('Depricated arguments!')

        # per awg:
        #    waveform_names
        #    waveforms
        
        
        count = len(gates)
        if count != len(waveforms):
            raise VirtualAwgError('Invalid number of gates/waveforms!')

        for index in range(count):
            gate = gates[index]
            sequences = waveforms[index]


        for gate, waveform in waveforms.items():
            (awg_nr, channel_nr, *marker_nr) = self.parameters.awg_map[gate]
            sampling_rate = self.awgs[awg_nr].get_sampling_rate()
            raw_wave = self.__get_raw_data(waveform, sampling_rate)
            


        # get awg_channels for each awg
        awg_channels = [set() for x in range(self.awg_count)]
        for gate, _ in waveforms.items():
            (awg_nr, channel_nr, *_) = self.parameters.awg_map[gate]
            awg_channels[awg_nr].add(channel_nr)
        awg_channels = [list(item) for item in awg_channels]


        # create and fill elements...
        element_count = max(len(seqs) for seqs in waveforms.values())
        elements = [[Element(awg_channels[awg_index], [1, 2]) for x in range(element_count)]
                             for awg_index in range(self.awg_count)]

        for gate, waveform in waveforms.items():
            (awg_nr, channel_nr, *marker_nr) = self.parameters.awg_map[gate]
            sampling_rate = self.awgs[awg_nr].get_sampling_rate()
            for wf in waveform:
                raw_wave = get_raw_data(waveform, sampling_rate)
                if not marker_nr:
                    elements[awg_nr][0].set_channel(channel_nr, raw_wave)
                elif marker_nr:
                    elements[awg_nr][0].set_marker(channel_nr, marker_nr[0], raw_wave)

        # fill other empty elements.
        for awg_nr, element_nr in itertools.product(range(awg_count), range(element_count)):
            elements[awg_nr][element_nr].finalize_element()

        # upload elements...
        if do_upload:
            for awg_nr in range(awg_count):
                self.awgs[awg_nr].reset_waveform_channels(awg_channels[awg_nr])
                self.awgs[awg_nr].set_sequences(elements[awg_nr])
                self.awgs[awg_nr].send_waveforms()

        #(names, waveforms, m1s, m2s, sequence,
        #nreps, trig_waits, goto_states, jump_tos, params):


    def sweep_gates(self, gate_s, period, marker_delay=0, width=0.95):
        """ Creates sawtooth waveforms for the given gates with marker channel for
            the digitizer.

        Arguments:
            gate_s (dict): A dictionary with a gate as key and sweep range Vpp as value.
            period (float): The duration of the sawtooth in seconds.
            marker_delay (float): The delay of the marker pulse in seconds. The marker_delay must always
                                  be less than the period.
            width (float): the width of the sawtooth.

        Returns:
            waveform_s (dict): A dictonary with the waveforms and sweep properties (width, period,
                              range, sample rate and marker delay)

        Example:
        -------
        >>> <VirtualAwg>.sweep_gates({'X1': 0.5, 'P2': 1.0}, period=10e-7)
        """
        if not self.are_awg_gates(list(gate_s.keys())):
            raise VirtualAwgError('Unusable gate selected!')

        period_in_ns = period*10**9
        waveform_s = {gate: make_sawtooth(vpp, period_in_ns) for (gate, vpp) in gate_s.items()}
        delay = int(marker_delay * self.parameters.sampling_rate() + 1)
        samples = int(period * self.parameters.sampling_rate() + 1)
        array = np.zeros(samples)
        array[delay: delay+samples//20] = 1.0
        waveform_s['dig_mk'] = {'TYPE': DataType.RAW_DATA, 'WAVE': array}

        self.sweep_init(waveform_s)
        self.sweep_run()

        waveform_s['width'] = width
        waveform_s['period'] = period
        waveform_s['sweeprange'] = list(gate_s.values())
        waveform_s['samplerate'] = self.parameters.sampling_rate()
        waveform_s['markerdelay'] = self.parameters.marker_delay()
        return waveform_s, None

    def sweep_gates_2D(self, gates_horz, gates_vert, sweepranges, resolution, width=.95, do_upload=True):
        pass

    def sweep_2D_process(self, data, waveform, diff_dir=None):
        pass

    def pulse_gates(self, gate_voltages, waittimes, ramp_params=None, delete=True):
        pass
    
    def sweep_process(self, data, waveform, averages=1, direction='forwards', start_offset=1):
        '''Process the returned data using shape of the sawtooth send with the AWG.'''
        pass

# -----------------------------------------------------------------------------


class VirtualAwgBase():

    def __init__(self, channels, markers):
        self.channel_count = len(channels)
        self.channels = channels
        self.marker_count = len(markers)
        self.markers = markers
        # TODO change to ab_class...

# -------------------------------------------------------------------------------------------------


class TektronixVirtualAwg(VirtualAwgBase):

    def __init__(self, awg):
        super().__init__(channels=[1, 2, 3, 4], markers=[1, 2])
        self.__awg = awg

    @property
    def get_base_awg(self):
        return self.__awg

    def set_continues_mode(self):
        self.__awg.set('run_mode', 'CONT')

    def set_sequence_mode(self):
        self.__awg.set('run_mode', 'SEQ')

    def set_sampling_rate(self, value=1e9):
        self.__awg.set('clock_freq', value)

    def get_sampling_rate(self):
        return self.__awg.get('clock_freq')

    def get_amplitude(self, channel):
        return self.__awg.get('ch{0}_amp'.format(channel))

    def set_awg_properties(self, parameters):
        for channel in range(1, 5):
            self.__awg.set('ch{0}_amp'.format(channel), parameters.channel_amplitudes())
            self.__awg.set('ch{0}_offset'.format(channel), parameters.channel_offset())
            self.__awg.set('ch{0}_m1_low'.format(channel), parameters.marker_low())
            self.__awg.set('ch{0}_m1_high'.format(channel), parameters.marker_high())
            self.__awg.set('ch{0}_m1_del'.format(channel), parameters.marker_delay())
            self.__awg.set('ch{0}_m2_low'.format(channel), parameters.marker_low())
            self.__awg.set('ch{0}_m2_high'.format(channel), parameters.marker_high())
            self.__awg.set('ch{0}_m2_del'.format(channel), parameters.marker_delay())

    def run(self):
        self.__awg.run()

    def set_outputs(self, is_enabled, channels=None):
        if not channels:
            channels = self.channels
        state = 1 if is_enabled else 0
        [self.__awg.set('ch{0}_state'.format(ch), state) for ch in channels]

    def stop(self):
        self.__awg.stop()

    def reset(self):
        self.__awg.reset()

    def set_sequence(self, channels, sequence, waits=None, repeats=None, gotos=None):
        if waits or repeats or gotos:
            raise NotImplementedError("Wait, repeats, gotos currently not implemented!")
        if not sequence or len(sequence) != len(channels):
            raise VirtualAwgError('Invalid sequence and channel count!')
        if not all(len(idx) == len(sequence[0]) for idx in sequence):
            raise VirtualAwgError('Invalid sequence list lengthts!')
        request_rows = len(sequence[0])
        current_rows = self.__get_sequence_length()
        if request_rows != current_rows:
            self.__set_sequence_length(request_rows)
        for row_index in range(request_rows):
            for channel in self.channels:
                if channel in channels:
                    ch_index = channels.index(channel)
                    wave_name = sequence[ch_index][row_index]
                    self.__awg.set_sqel_waveform(wave_name, channel, row_index + 1)
                else:
                    self.__awg.set_sqel_waveform("", channel, row_index + 1)
        self.__awg.set_sqel_goto_state(request_rows, 1)

    def delete_sequence(self):
            self.__set_sequence_length(0)

    def upload_waveforms(self, names, waveforms, params):
        pack_count = len(names)
        packed_waveforms = dict()
        [wfs, m1s, m2s] = list(map(list, zip(*waveforms)))
        for i in range(pack_count):
            name = names[i]
            package = self.__awg.pack_waveform(wfs[i], m1s[i], m2s[i])
            packed_waveforms[name] = package

        amplitude = params.channel_amplitudes()
        offset = params.channel_offset()
        marker_low = params.marker_low()
        marker_high = params.marker_high()
        channel_cfg = {'ANALOG_METHOD_1': 1, 'CHANNEL_STATE_1': 1, 'ANALOG_AMPLITUDE_1': amplitude,
                       'ANALOG_OFFSET_1': offset,
                       'MARKER1_METHOD_1': 2, 'MARKER1_LOW_1': marker_low, 'MARKER1_HIGH_1': marker_high,
                       'MARKER2_METHOD_1': 2, 'MARKER2_LOW_1': marker_low, 'MARKER2_HIGH_1': marker_high,

                       'ANALOG_METHOD_2': 1, 'CHANNEL_STATE_2': 1, 'ANALOG_AMPLITUDE_2': amplitude,
                       'ANALOG_OFFSET_2': offset,
                       'MARKER1_METHOD_2': 2, 'MARKER1_LOW_2': marker_low, 'MARKER1_HIGH_2': marker_high,
                       'MARKER2_METHOD_2': 2, 'MARKER2_LOW_2': marker_low, 'MARKER2_HIGH_2': marker_high,

                       'ANALOG_METHOD_3': 1, 'CHANNEL_STATE_3': 1, 'ANALOG_AMPLITUDE_3': amplitude,
                       'ANALOG_OFFSET_3': offset,
                       'MARKER1_METHOD_3': 2, 'MARKER1_LOW_3': marker_low, 'MARKER1_HIGH_3': marker_high,
                       'MARKER2_METHOD_3': 2, 'MARKER2_LOW_3': marker_low, 'MARKER2_HIGH_3': marker_high,

                       'ANALOG_METHOD_4': 1, 'CHANNEL_STATE_4': 1, 'ANALOG_AMPLITUDE_4': amplitude,
                       'ANALOG_OFFSET_4': offset,
                       'MARKER1_METHOD_4': 2, 'MARKER1_LOW_4': marker_low, 'MARKER1_HIGH_4': marker_high,
                       'MARKER2_METHOD_4': 2, 'MARKER2_LOW_4': marker_low, 'MARKER2_HIGH_4': marker_high}

        file_name = 'costum_awg_file.awg'
        self.__awg.visa_handle.write('MMEMory:CDIRectory "C:\\Users\\OEM\\Documents"')
        awg_file = self.__awg.generate_awg_file(packed_waveforms, np.array([]), [], [], [], [], channel_cfg)
        self.__awg.send_awg_file(file_name, awg_file)
        current_dir = self.__awg.visa_handle.query('MMEMory:CDIRectory?')
        current_dir = current_dir.replace('"', '')
        current_dir = current_dir.replace('\n', '\\')
        self.__awg.load_awg_file('{0}{1}'.format(current_dir, file_name))

    def delete_waveforms(self):
        self.__awg.delete_all_waveforms_from_list()

    def __set_sequence_length(self, count):
        self.__awg.write('SEQuence:LENGth {0}'.format(count))

    def __get_sequence_length(self):
        row_count = self.__awg.ask('SEQuence:LENGth?')
        return int(row_count)

    def __get_rescaled_waveform(self, channel, waveform):
        amplitude = self.get_amplitude(channel)
        if any(abs(item) > amplitude for item in waveform):
            logging.error('Waveform contains invalid values! Will set items to zero.')
        return [(lambda value: 0 if value > amplitude else value/amplitude)(value) for value in waveform]

# -----------------------------------------------------------------------------------------------


class KeysightVirtualAwg(VirtualAwgBase):

    def __init__(self, awg):
        super().__init__(channels=[1,2,3,4], markers=[1])
        self.__awg = awg

# -----------------------------------------------------------------------------------------------


class VirtualAwgError(Exception):
    '''Exception for a specific error related to the virual AWG.'''