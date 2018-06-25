import numpy as np
from matplotlib import pyplot as plt

from qctoolkit.pulses import SequencePT, TablePT
from qctoolkit.pulses.plotting import (PlottingNotPossibleException, plot, render)
from qctoolkit.pulses.sequencing import Sequencer as Sequencing
from qctoolkit.serialization import Serializer, DictBackend
from qtt.instrument_drivers.virtual_awg.templates import DataTypes, Templates
from qtt.instrument_drivers.virtual_awg.serializer import StringBackend


class Sequencer:

    """Conversion factor from seconds to nano-seconds."""
    __sec_to_ns = 1e-9

    @staticmethod
    def __qc_toolkit_template_to_array(sequence, sampling_rate):
        """Renders a QC toolkit sequence as array with voltages.

        Arguments:
            sequence (dict): a waveform is a dictionary with "type" value
            given the used pulse library. The "wave" value should contain
            the actual wave-object.
            sampling_rate (float): The number of samples per second.

        Returns:
            voltages (np.array): The array with voltages generated from the template.
        """
        sequencer = Sequencing()
        template = sequence['WAVE']
        channels = template.defined_channels
        sequencer.push(template, dict(), channel_mapping={ch: ch for ch in channels},
                       window_mapping={w: w for w in template.measurement_names})
        instructions = sequencer.build()
        if not sequencer.has_finished():
            raise PlottingNotPossibleException(template)
        (_, voltages) = render(instructions, sampling_rate * Sequencer.__sec_to_ns)
        return voltages[next(iter(voltages))]

    @staticmethod
    def __raw_data_to_array(sequence, sampling_rate):
        """Renders a raw sequence as array with voltages.

        Arguments:
            sequence (dict): a waveform is a dictionary with "type" value
            given the used pulse library. The "wave" value should contain
            the actual wave-object.
            sampling_rate (float): The number of samples per second.

        Returns:
            voltages (np.array): The array with voltages generated from the template.
        """
        return sequence['WAVE']

    @staticmethod
    def get_data(sequence, sampling_rate):
        """This function returns the raw array data given a sequence.
           A sequence can hold different types of data dependend on the
           used pulse library. Currently only raw array data and QC-toolkit
           can be used.

        Arguments:
            sequence (dict): a waveform is a dictionary with "type" value
            given the used pulse library. The "wave" value should contain
            the actual wave-object.
            sampling_rate: a sample rate of the awg in samples per sec.

        Returns:
            A numpy.ndarray with the corresponding sampled voltages.
        """
        data_type = sequence['TYPE']
        switch = {DataTypes.RAW_DATA: Sequencer.__raw_data_to_array,
                  DataTypes.QC_TOOLKIT: Sequencer.__qc_toolkit_template_to_array}
        to_array_function = switch[data_type]
        return to_array_function(sequence, sampling_rate)

    @staticmethod
    def __raw_data_plot(sequence, sampling_rate):
        """Plots a raw data sequence."""
        raw_data = Sequencer.__raw_data_to_array(sequence, sampling_rate)
        sample_count = len(raw_data)
        total_time = (sample_count - 1)/(sampling_rate * Sequencer.__sec_to_ns)
        times = np.linspace(0, total_time, num=sample_count, dtype=float)
        plt.step(times, raw_data, where='post')
        plt.show()

    @staticmethod
    def __qc_toolkit_template_plot(sequence, sampling_rate):
        """Plots a QC toolkit sequence."""
        ns_sample_rate = sampling_rate * Sequencer.__sec_to_ns
        plot(sequence['WAVE'], sample_rate=ns_sample_rate)
        plt.show()

    @staticmethod
    def plot(sequence, sampling_rate):
        """Creates a plot for viewing the sequence."""
        data_type = sequence['TYPE']
        switch = {DataTypes.RAW_DATA: Sequencer.__raw_data_plot,
                  DataTypes.QC_TOOLKIT: Sequencer.__qc_toolkit_template_plot}
        plot_function = switch[data_type]
        plot_function(sequence, sampling_rate)

    @staticmethod
    def __raw_data_serialize(sequence):
        pass

    @staticmethod
    def __qc_toolkit_serialize(sequence):
        backend = DictBackend()
        serializer = Serializer(backend)
        return serializer.serialize(sequence, overwrite=True)

    @staticmethod
    def serialize(sequence):
        data_type = sequence['TYPE']
        switch = {DataTypes.RAW_DATA: Sequencer.__raw_data_serialize,
                  DataTypes.QC_TOOLKIT: Sequencer.__qc_toolkit_serialize}
        serialize_function = switch[data_type]
        return serialize_function(sequence['WAVE'])

    @staticmethod
    def deserialize(json_string):
        backend = DictBackend()
        serializer = Serializer(backend)
        return serializer.deserialize(json_string)

    @staticmethod
    def make_sawtooth_wave(amplitude, period, repetitions=1, name='sawtooth'):
        """Creates a sawtooth waveform of the type QC toolkit template.

        Arguments:
            amplitude (float): The peak-to-peak voltage of the waveform.
            period (float): The period of the waveform in seconds.
            repetitions (int): The number of oscillations in the sequence.
            name (str): The name of the returned sequence.

        Returns:
           (dict) A dictionary with name, type and template of the waveform.
        """
        seq_data = (Templates.sawtooth(name), {'period': period*1e9, 'amplitude': amplitude/2})
        return {'NAME': name, 'WAVE': SequencePT(*((seq_data,)*repetitions)), 'TYPE': DataTypes.QC_TOOLKIT}

    @staticmethod
    def make_square_wave(amplitude, period, repetitions=1, name='pulse'):
        """Creates a block waveforms of the type QC toolkit template.

        Arguments:
            amplitude (float): The peak-to-peak voltage of the waveform.
            period (float): The period of the waveform in seconds.
            repetitions (int): The number of oscillations in the sequence.
            name (str): The name of the returned sequence.

        Returns:
            (dict): A dictionary with name, type and template of the waveform.
        """
        seq_data = (Templates.square(name), {'period': period*1e9, 'amplitude': amplitude})
        return {'NAME': name, 'WAVE': SequencePT(*(seq_data,)*repetitions), 'TYPE': DataTypes.QC_TOOLKIT}

    @staticmethod
    def make_marker(period, repetitions=1, uptime=0.2, name='marker'):
        seq_data = (Templates.marker(name), {'period': period*1e9, 'uptime': uptime})
        return {'NAME': name, 'WAVE': SequencePT(*((seq_data,)*repetitions)), 'TYPE': DataTypes.QC_TOOLKIT}

# UNITTESTS #

def test_make_sawtooth_HasCorrectProperties():
    period = 1e-6
    amplitude = 1.5
    sampling_rate = 1e9
    sawtooth_sequence = Sequencer.make_sawtooth_wave(amplitude, period=period)
    raw_data = Sequencer.get_data(sawtooth_sequence, sampling_rate)
    assert(np.max(raw_data) == amplitude/2)
    assert(np.min(raw_data) == -amplitude/2)
    assert(len(raw_data) == sampling_rate*period+1)


def test_serializer():
    sawtooth = Sequencer.make_sawtooth_wave(1.5, 1e-6, repetitions=1)
    return Sequencer.serialize(sawtooth)
