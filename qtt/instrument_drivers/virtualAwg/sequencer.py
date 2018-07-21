import numpy as np
from matplotlib import pyplot as plt

from qctoolkit.pulses import SequencePT
from qctoolkit.pulses.plotting import (PlottingNotPossibleException, plot, render)
from qctoolkit.pulses.sequencing import Sequencer as Sequencing
from qctoolkit.serialization import Serializer, DictBackend
from qtt.instrument_drivers.virtualawg.templates import DataTypes, Templates
# from qtt.instrument_drivers._virtual_awg.serializer import StringBackend


class Sequencer:

    """Conversion factor from seconds to nano-seconds."""
    __sec_to_ns = 1e9

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
        template = sequence['wave']
        channels = template.defined_channels
        sequencer.push(template, dict(), channel_mapping={ch: ch for ch in channels},
                       window_mapping={w: w for w in template.measurement_names})
        instructions = sequencer.build()
        if not sequencer.has_finished():
            raise PlottingNotPossibleException(template)
        (_, voltages) = render(instructions, sampling_rate / Sequencer.__sec_to_ns)
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
        return sequence['wave']

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
            sampling_rate (float): a sample rate of the awg in samples per sec.

        Returns:
            A numpy.ndarray with the corresponding sampled voltages.
        """
        data_type = sequence['type']
        switch = {DataTypes.RAW_DATA: Sequencer.__raw_data_to_array,
                  DataTypes.QC_TOOLKIT: Sequencer.__qc_toolkit_template_to_array}
        to_array_function = switch[data_type]
        return to_array_function(sequence, sampling_rate)

    @staticmethod
    def __raw_data_plot(sequence, sampling_rate, axes):
        """Plots a raw data sequence."""
        if axes is None:
            figure = plt.figure()
            axes = figure.add_subplot(111)
        raw_data = Sequencer.__raw_data_to_array(sequence, sampling_rate)
        sample_count = len(raw_data)
        total_time = (sample_count - 1)/(sampling_rate * Sequencer.__sec_to_ns)
        times = np.linspace(0, total_time, num=sample_count, dtype=float)
        axes.step(times, raw_data, where='post')
        axes.get_figure().show()

    @staticmethod
    def __qc_toolkit_template_plot(sequence, sampling_rate, axes):
        """Plots a QC toolkit sequence."""
        ns_sample_rate = sampling_rate / Sequencer.__sec_to_ns
        plot(sequence['wave'], sample_rate=ns_sample_rate, axes=axes, show=False)

    @staticmethod
    def plot(sequence, sampling_rate, axes=None):
        """Creates a plot for viewing the sequence.

        Arguments:
            sequence (dict): a waveform is a dictionary with "type" value
            given the used pulse library. The "wave" value should contain
            the actual wave-object.
            sampling_rate (float): a sample rate of the awg in samples per sec.
            axes: matplotlib Axes object the pulse will be drawn into if provided.
        """
        data_type = sequence['type']
        switch = {DataTypes.RAW_DATA: Sequencer.__raw_data_plot,
                  DataTypes.QC_TOOLKIT: Sequencer.__qc_toolkit_template_plot}
        plot_function = switch[data_type]
        plot_function(sequence, sampling_rate, axes)

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
        data_type = sequence['type']
        switch = {DataTypes.RAW_DATA: Sequencer.__raw_data_serialize,
                  DataTypes.QC_TOOLKIT: Sequencer.__qc_toolkit_serialize}
        serialize_function = switch[data_type]
        return serialize_function(sequence['wave'])

    @staticmethod
    def deserialize(json_string):
        backend = DictBackend()
        serializer = Serializer(backend)
        return serializer.deserialize(json_string)

    @staticmethod
    def make_sawtooth_wave(amplitude, period, width=0.95, repetitions=1, name='sawtooth'):
        """Creates a sawtooth waveform of the type QC toolkit template.

        Arguments:
            amplitude (float): The peak-to-peak voltage of the waveform.
            period (float): The period of the waveform in seconds.
            repetitions (int): The number of oscillations in the sequence.
            name (str): The name of the returned sequence.

        Returns:
           (dict) A dictionary with name, type and template of the waveform.
        """
        if width <= 0 or width >= 1:
            raise ValueError('Invalid argument value (0 < width < 1)!')
        input_variables = {'period': period*Sequencer.__sec_to_ns, 'amplitude': amplitude/2.0,
                           'width': width}
        sequence_data = (Templates.sawtooth(name), input_variables)
        return {'name': name, 'wave': SequencePT(*((sequence_data,)*repetitions)),
                'type': DataTypes.QC_TOOLKIT}

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
        input_variables = {'period': period*Sequencer.__sec_to_ns, 'amplitude': amplitude/2.0}
        sequence_data = (Templates.square(name), input_variables)
        return {'name': name, 'wave': SequencePT(*(sequence_data,)*repetitions),
                'type': DataTypes.QC_TOOLKIT}

    @staticmethod
    def make_marker(period, uptime=0.2, offset=0.0, repetitions=1, name='marker'):
        """Creates a marker block waveforms of the type QC toolkit template.

        Arguments:
            period (float): The period of the waveform in seconds.
            uptime (float): The marker up period in seconds.
            offset (float): The marker delay in seconds.
            repetitions (int): The number of oscillations in the sequence.
            name (str): The name of the returned sequence.

        Returns:
            (dict): A dictionary with name, type and template of the waveform.
        """
        if uptime <= 0 or offset < 0:
            raise ValueError('Invalid argument value (uptime <= 0 or offset < 0)!')
        if uptime + offset > period:
            raise ValueError('Invalid argument value (uptime + offset > period)!')
        input_variables = {'period': period*Sequencer.__sec_to_ns,
                           'uptime': uptime*Sequencer.__sec_to_ns,
                           'offset': offset*Sequencer.__sec_to_ns}
        sequence_data = (Templates.marker(name), input_variables)
        return {'name': name, 'wave': SequencePT(*((sequence_data,)*repetitions)),
                'type': DataTypes.QC_TOOLKIT}

# UNITTESTS #


def test_qc_toolkit_sawtooth_HasCorrectProperties():
    epsilon = 1e-14
    period = 1e-3
    amplitude = 1.5
    sampling_rate = 1e9
    sequence = Sequencer.make_sawtooth_wave(amplitude, period)
    raw_data = Sequencer.get_data(sequence, sampling_rate)
    assert len(raw_data) == sampling_rate*period + 1
    assert np.abs(np.min(raw_data) + amplitude/2) <= epsilon
    assert np.abs(np.max(raw_data) - amplitude/2) <= epsilon


def test_raw_wave_HasCorrectProperties():
    epsilon = 1e-14
    period = 1e-3
    amplitude = 1.5
    sampling_rate = 1e9
    name = 'test_raw_data'
    sequence = {'name': name, 'wave': [0]*int(period*sampling_rate+1),
                'type': DataTypes.RAW_DATA}
    raw_data = Sequencer.get_data(sequence, sampling_rate)
    assert len(raw_data) == sampling_rate*period+1
    assert np.min(raw_data) == 0


def test_serializer():
    period = 1e-6
    amplitude = 1.5
    sawtooth = Sequencer.make_sawtooth_wave(amplitude, period)
    return Sequencer.serialize(sawtooth)
