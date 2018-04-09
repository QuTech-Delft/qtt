import logging
import itertools
import numpy as np

from enum import Enum
from qcodes.instrument.base import InstrumentBase

from qctoolkit.pulses import SequencePT, TablePT
from qctoolkit.pulses.sequencing import Sequencer
from qctoolkit.pulses.plotting import plot, render, PlottingNotPossibleException

# -------------------------------------------------------------------------------------------------


def pulsewave_template(name: str='pulse'):
    return TablePT({name: [(0, 'amplitude'), ('width', 0), ('holdtime', 0)]})


def sawtooth_template(name: str='sawtooth'):
    return TablePT({name: [(0, 0), ('period/4', 'amplitude', 'linear'),
                           ('period*3/4', '-amplitude', 'linear'),
                           ('period', 0, 'linear')]})


def wait_template(name: str='wait'):
    return TablePT({name: [(0, 0), ('holdtime', 0)]})


def to_array(template, samples_per_ns):
    """ Renders the QC toolkit template as voltages array.

    Arguments:
        template (*PT): A QC Toolkit template of type; pulsePT,
                        functionPT, pointPT or sequencePT.
        samples_per_ns (float): The number of samples per nanosecond.

    Returns:
        voltages (np.array): The array with voltages generated from
                             the template.
    """
    channels = template.defined_channels
    parameters = dict()
    sequencer = Sequencer()
    sequencer.push(template, parameters,
                   channel_mapping={ch: ch for ch in channels},
                   window_mapping={w: w for w in template.measurement_names})
    sequence = sequencer.build()
    if not sequencer.has_finished():
        raise PlottingNotPossibleException(template)
    (_, voltages) = render(sequence, samples_per_ns)
    return voltages[next(iter(voltages))]  # ugly!!!

# -----------------------------------------------------------------------------


class DataType(Enum):
    RAW_DATA = 0
    QC_TOOLKIT = 1


def make_sawtooth(vpp, period, repetitions=1):
    values = {'period': period, 'amplitude': vpp/2}
    data = (sawtooth_template(), values)
    return {'TYPE': DataType.QC_TOOLKIT, 'WAVE':
            SequencePT(*((data,)*repetitions))}


def make_pulses(voltages, waittimes, filter_cutoff=None, mvrange=None):
    sequence = []
    return sequence


def plot_waveform(waveform, sample_rate: int):
    """ Plots the waveform array."""
    from matplotlib import pyplot as plt
    sample_count = len(waveform)
    total_time = sample_count / sample_rate
    times = np.linspace(0, total_time, num=sample_count)
    plt.plot(times, waveform)


def plot_sequence(sequence, sample_rate):
    """ Plots the qc-toolkit sequence."""
    plot(sequence['WAVE'], sample_rate=sample_rate)

# -------------------------------------------------------------------------------------------------


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
        awg_count = len(self.awgs)
        if awg_count == 0:
            logging.warning("No physical awg's connected!")
            return
        if awg_count == 1:
            self.awgs[0].set_sequence_mode()
            logging.info("One physical awg's connected!")
        elif awg_count == 2:
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

    def stop_awgs(self):
        """ Stops all awg(s) and turns off all channels."""
        [awg.stop() for awg in self.awgs]
        logging.info("All awg's stopped...")

    def reset_awgs(self):
        """ Resets all awg(s) and turns of all channels."""
        [awg.reset() for awg in self.awgs]
        [awg.set_awg_properties(self.parameters) for awg in self.awgs]
        logging.info("All awg's are reseted...")

    def __get_raw_data(self, waveform, sampling_rate):
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
            return to_array(waveform['WAVE'], sampling_rate/10**9)

    def sweep_init(self, waveforms, do_upload=True, period=None, delete=None, samp_freq=None):
        """ Sends the waveform(s) to gate(s) and markers(s).

        Arguments:
            waveforms (dict): the waveforms with the gates/markers as keys. Note markers must have
                              waveforms with only 0 and 1's.
            do_upload (bool): indicates whether the waveforms are uploaded.
            period, delete, sample_freq: depricated arguments. To be removed later!

        Example:
        -------
        >>> <VirtualAwg>.sweep_init(waveforms)
        """
        if period or delete or samp_freq:
            logging.error('Arguments: period, delete, samp_rate are depricated!')
            raise VirtualAwgError('Depricated arguments!')

        # get awg_channels for each awg
        awg_count = len(self.awgs)
        awg_channels = [set() for x in range(awg_count)]
        for gate, _ in waveforms.items():
            (awg_nr, channel_nr, *_) = self.parameters.awg_map[gate]
            awg_channels[awg_nr].add(channel_nr)
        awg_channels = [list(item) for item in awg_channels]

        # create and fill elements...
        element_count = 1  # max(len(seqs) for seqs in waveforms.values())
        elements = [[Element(awg_channels[awg_index], [1, 2]) for x in range(element_count)] 
                             for awg_index in range(len(self.awgs))]
        for gate, waveform in waveforms.items():
            (awg_nr, channel_nr, *marker_nr) = self.parameters.awg_map[gate]
            #for wave in waveform:
            sampling_rate = self.awgs[awg_nr].get_sampling_rate()
            raw_wave = self.__get_raw_data(waveform, sampling_rate)
            if not marker_nr:
                elements[awg_nr][0].set_channel(channel_nr, raw_wave)
            else:
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

    def sweep_run(self):
        """ Turns on the AWG channels selected using the gates. Used after sweep_init."""
        [awg.run() for awg in self.awgs]

    def sweep_gates(self, gate_s, period, marker_delay, width=0.95):

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

    def pulse_gates(self, gate_voltages, waittimes, filtercutoff=None, delete=True):
        pass

    def sweep_process(self, data, waveform, averages=1, direction='forwards', start_offset=1):
        '''Process the returned data using shape of the sawtooth send with the AWG.'''
        pass

# -----------------------------------------------------------------------------


class VirtualAwgBase():

    channels = 4
    markers = 2

    def set_master(self):
        pass

    def set_slave(self):
        pass

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

# -------------------------------------------------------------------------------------------------


class TektronixVirtualAwg(VirtualAwgBase):

    def __init__(self, awg):
        self.__awg = awg

    @property
    def get(self):
        return self.__awg

    def set_continues_mode(self):
        self.__awg.set('run_mode', 'CONT')

    def set_sequence_mode(self):
        self.__awg.set('run_mode', 'SEQ')
        # self.__awg.sequence_length.set(1)
        # self.__awg.set_sqel_trigger_wait(1, 0)

    def set_sampling_rate(self, value=1e9):
        self.__awg.set('clock_freq', value)

    def get_sampling_rate(self):
        return self.__awg.get('clock_freq')

    def get_amplitude(self, channel):
        return self.__awg.get('ch{0}_amp'.format(channel))

    def set_awg_properties(self, parameters):
        self.__awg.set('clock_freq', parameters.sampling_rate())
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
        [self.__awg.set('ch{0}_state'.format(ch), 1) for ch in self.waveform_channels]
        self.__awg.run()

    def stop(self):
        [self.__awg.set('ch{0}_state'.format(ch), 0) for ch in self.waveform_channels]
        self.__awg.stop()

    def reset(self):
        self.__awg.reset()

    def delete_all_waveforms(self):
        self.__awg.delete_all_waveforms_from_list()

    def reset_waveform_channels(self, channels=[1, 2, 3, 4]):
        self.waveform_channels = channels
        self.marker_channels = [1, 2]
        self.waveforms = [[] for x in range(len(channels))]
        self.marker1s = [[] for x in range(len(channels))]
        self.marker2s = [[] for x in range(len(channels))]
        self.jump_tos = []
        self.nreps = []

    def set_sequences(self, elements, repeats=1):
        for element in elements:
            for ch_index in range(len(self.waveform_channels)):
                self.waveforms[ch_index].append(element.waveforms[ch_index])
                self.marker1s[ch_index].append(element.markers1[ch_index])
                self.marker2s[ch_index].append(element.markers2[ch_index])
            self.jump_tos.append(len(self.jump_tos) + 2)
            self.nreps.append(repeats)

    def send_waveforms(self):
        data_count = len(self.nreps)
        assert(data_count != 0)
        goto_states = [0]*data_count
        trigger_waits = [0]*data_count
        self.jump_tos[-1] = 1
        self.__awg.make_send_and_load_awg_file(self.waveforms, self.marker1s, self.marker2s,
                                               self.nreps, trigger_waits, self.jump_tos, goto_states,
                                               self.waveform_channels)

    #def add_sequence_index(self, element, repeats=1):
    #    for ch_index in range(len(self.waveform_channels)):
    #        self.waveforms[ch_index].append(element.waveform[ch_index])
    #        self.marker1s[ch_index].append(element.marker1[ch_index])
    #        self.marker2s[ch_index].append(element.marker2[ch_index])
    #    self.jump_tos.append(len(self.jump_tos) + 2)
    #    self.nreps.append(repeats)

    def __get_rescaled_waveform(self, channel, waveform):
        amplitude = self.get_amplitude(channel)
        if any(abs(item) > amplitude for item in waveform):
            logging.error('Waveform contains invalid values! Will set items to zero.')
        return [(lambda value: 0 if value > amplitude else value/amplitude)(value) for value in waveform]


class Element:

    def __init__(self, channels, markers):
        channel_count = len(channels)
        self.channels = channels
        self.markers = markers  # TODO!!!
        self.waveforms = [None]*channel_count
        self.markers1 = [None]*channel_count
        self.markers2 = [None]*channel_count

    def set_channel(self, channel, waveform):
        index = self.channels.index(channel)
        self.waveforms[index] = waveform

    def set_marker(self, channel, marker, waveform):
        index = self.channels.index(channel)
        if marker == 1:
            self.markers1[index] = waveform
        elif marker == 2:
            self.markers2[index] = waveform

    def finalize_element(self):
        wave_data_count = next((len(w) for w in self.waveforms if w is not None), 0)
        marker1_data_count = next((len(m1) for m1 in self.markers1 if m1 is not None), 0)
        marker2_data_count = next((len(m2) for m2 in self.markers2 if m2 is not None), 0)
        data_count = max(wave_data_count, marker1_data_count, marker2_data_count)
        self.__set_element_items('Waveform', self.waveforms, data_count)
        self.__set_element_items('Marker1', self.markers1, data_count)
        self.__set_element_items('Marker2', self.markers2, data_count)

    def __set_element_items(self, name, data_list, max_data_count):
        for index in range(len(data_list)):
            if data_list[index] is None:
                data_list[index] = np.zeros(max_data_count)
            elif len(data_list[index]) != max_data_count:
                logging.error('{0} lengths are unequal!'.format(name))
                raise VirtualAwgError('{0} lengths are unequal!'.format(name))


# -------------------------------------------------------------------------------------------------


class KeysightVirtualAwg(VirtualAwgBase):
    pass

# -------------------------------------------------------------------------------------------------


class VirtualAwgError(Exception):
    '''Exception for a specific error related to the virual AWG.'''