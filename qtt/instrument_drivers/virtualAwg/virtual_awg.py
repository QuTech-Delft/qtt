import logging
import numpy as np

from qcodes import Instrument
from qtt.instrument_drivers.virtualAwg.sequencer import Sequencer
from qtt.instrument_drivers.virtualAwg.awgs.Tektronix5014C import Tektronix5014C_AWG

from qtt.instrument_drivers.virtualAwg.awgs.KeysightM3202A import KeysightM3202A_AWG
from qtt.instrument_drivers.virtualAwg.awgs.ZurichInstrumentsHDAWG8 import ZurichInstrumentsHDAWG8


class VirtualAwgError(Exception):
    """ Exception for a specific error related to the virtual AWG."""


class VirtualAwg(Instrument):
    """ The virtual AWG is an abstraction layer to produce pulse driven state manipulation of physical qubits.
        The class aims for hardware independent control, where only common arbitrary waveform generator (AWG)
        functionality is used. A translation between the AWG channels and the connected quantum gates provide
        the user control of the AWG's in terms of gate names and waveform sequences only. The virtual AWG is
        used for fast change of the DC landscape by changing the voltage levels of the gates. No microwave
        control is involved, meaning not related to the spin degrees of freedom of the qubits.
    """

    __digitizer_name = 'm4i_mk'
    __awg_slave_name = 'awg_mk'
    __volt_to_millivolt = 1e-3

    def __init__(self, awgs, settings, name='virtual_awg', logger=logging, **kwargs):
        """ Creates and initializes an virtual AWG object and sets the relation between the quantum gates,
            markers and the AWG channels. The default settings (marker delays) are constructed at startup.

        Arguments:
            awgs (list): A list with AWG instances. Currently the following AWG's are supported:
                         Tektronix5014C, KeysightM3202A and the ZurichInstrumentsHDAWG8.
            settings (Instument): A class containing the settings of the quantum device, which are the
                                  awg_map (specificies the relation between the quantum gates, marker outputs
                                  and AWG channels) and the awg_to_plunger (specifies the attenuation factor between
                                  the AWG output and the voltage on the plunger).
        """
        super().__init__(name, **kwargs)
        self._settings = settings
        self._logger = logger
        self.__set_hardware(awgs)
        self.__set_parameters()

    def _get_virtual_info(self):
        """ Returns the data needed for snapshot of instrument."""
        return {'awg_map': self._settings.awg_map, 'awgs': [type(awg).__name__ for awg in self.awgs]}

    def __set_hardware(self, awgs):
        """ Sets the virtual AWG backends using the QCoDeS driver. Currently the following AWG are supported:
            the Tektronix AWG5014, the Keysight M3201A and the Zurich Instruments HDAWG8.

        Arguments:
            awgs (list): A list with the QCoDeS driver instances.
        """
        self.awgs = list()
        for awg in awgs:
            if type(awg).__name__ == 'Tektronix_AWG5014':
                self.awgs.append(Tektronix5014C_AWG(awg))
            elif type(awg).__name__ == 'Keysight_M3201A':
                self.awgs.append(KeysightM3202A_AWG(awg))
            elif type(awg).__name__ == 'ZIHDAWG8':
                self.awgs.append(ZurichInstrumentsHDAWG8(awg))
            else:
                raise VirtualAwgError('Unusable device added!')
        self.__awg_range = range(0, len(self.awgs))
        self.__awg_count = len(self.awgs)

    def __set_parameters(self):
        """ Constructs the parameters needed for setting the marker output settings for
            triggering the digitizer readout and starting slave AWG's when running a sequence.
        """
        if VirtualAwg.__awg_slave_name in self._settings.awg_map:
            self.add_parameter('awg_marker_delay', initial_value=0, set_cmd=None)
            self.add_parameter('awg_marker_uptime', initial_value=1e-8, set_cmd=None)
        if VirtualAwg.__digitizer_name in self._settings.awg_map:
            self.add_parameter('digitizer_marker_delay', initial_value=0, set_cmd=None)
            self.add_parameter('digitizer_marker_uptime', initial_value=1e-8, set_cmd=None)

    def run(self):
        """ Enables the main output of the AWG's."""
        _ = [awg.run() for awg in self.awgs]

    def stop(self):
        """ Disables the main output of the AWG's."""
        _ = [awg.stop() for awg in self.awgs]

    def reset(self):
        """ Resets all AWG's to its initialization state."""
        _ = [awg.reset() for awg in self.awgs]

    def enable_outputs(self, gate_names):
        """ Sets the given gates output to enabled. The gate map translates the given gate
            names to the correct AWG and channels. The digitizer and awg marker channels
            are automatically enabled if the channels are provided by the setttings awg_map.
            A start command is required to enable the outputs.

        Arguments;
            gate_names (list): The names of the gates which needs to be enabled.
        """
        if VirtualAwg.__digitizer_name in self._settings.awg_map:
            gate_names.extend([VirtualAwg.__digitizer_name])
        if VirtualAwg.__awg_slave_name in self._settings.awg_map:
            gate_names.extend([VirtualAwg.__awg_slave_name])
        for name in gate_names:
            (awg_number, channel_number, *_) = self._settings.awg_map[name]
            self.awgs[awg_number].enable_outputs([channel_number])

    def disable_outputs(self, gate_names):
        """ Sets the given gates output to disabled. The gate map translates the given gate
            names to the correct AWG and channels. The digitizer and awg marker channels
            are automatically disabled if the channels are provided by the setttings awg_map.
            A start command is required to enable the outputs.

        Arguments:
            gate_names (list) The names of the gates which needs to be disabled.
        """
        if VirtualAwg.__digitizer_name in self._settings.awg_map:
            gate_names.extend([VirtualAwg.__digitizer_name])
        if VirtualAwg.__awg_slave_name in self._settings.awg_map:
            gate_names.extend([VirtualAwg.__awg_slave_name])
        for name in gate_names:
            (awg_number, channel_number, *_) = self._settings.awg_map[name]
            self.awgs[awg_number].disable_outputs([channel_number])

    def update_setting(self, awg_number, setting, value):
        """ Updates a setting of the underlying AWG. The default settings are set
            during constructing of the AWG.

        Arguments:
            awg_number (int): The AWG number for the settings that will be changed.
            setting (str): The name of the setting e.g. 'amplitude'.
            value (float, int or string): The value of the setting e.g. 2.0 V.
        """
        if awg_number not in self.__awg_range:
            raise VirtualAwgError('Invalid AWG number {}!'.format(awg_number))
        self.awgs[awg_number].change_setting(setting, value)

    def are_awg_gates(self, gate_names):
        """ Checks whether the given quantum chip gates are connected to an AWG channel.

        Arguments:
            gate_names (str or list): the name(s) of the gates which needs to be checked.

        Returns:
            True if the gate or all gates are connected, else False.
        """
        if gate_names is None:
            return False
        if isinstance(gate_names, list):
            return np.all([self.are_awg_gates(g) for g in gate_names])
        if VirtualAwg.__digitizer_name or VirtualAwg.__awg_slave_name in gate_names:
            return False
        return True if gate_names in self._settings.awg_map else False

    def __make_markers(self, period, repetitions=1):
        """ Constructs the markers for triggering the digitizer readout and the slave AWG
            start sequence. The sequence length equals the period x repetitions.

        Arguments:
            period (float): The period of the markers in seconds.
            repetitions (int): The number of markers in the sequence.
        """
        marker_properies = dict()
        if VirtualAwg.__digitizer_name in self._settings.awg_map:
            digitizer_marker = Sequencer.make_marker(period, self.digitizer_marker_uptime(),
                                                     self.digitizer_marker_delay(), repetitions)
            marker_properies[VirtualAwg.__digitizer_name] = digitizer_marker
        if VirtualAwg.__awg_slave_name in self._settings.awg_map:
            awg_marker = Sequencer.make_marker(period, self.awg_marker_uptime(),
                                               self.awg_marker_delay(), repetitions)
            marker_properies[VirtualAwg.__awg_slave_name] = awg_marker
        return marker_properies

    def pulse_gates(self, gates, sweep_range, period, do_upload=True):
        """ Supplies a square wave to the given gates and returns the settings required
            for processing and constucting the readout times for the digitizer.

        Arguments:
            gates (dict): Contains the gate name keys with relative amplitude values.
            sweep_range (float): The overall amplitude of the pulse waves in millivolt.
            period (float): The period of the pulse waves in seconds.
            do_upload (bool, Optional): Does not upload the waves to the AWG's when set to False.

        Returns:
            A dictionary with the properties of the pulse waves; the original pulse sequence,
            the sweep ranges and the marker properties and period of the pulse waves.

        Example:
            >>> sec_period = 1e-6
            >>> mV_sweep_range = 100
            >>> gates = {'P4': 1, 'P7': 0.1}
            >>> pulse_data = virtualawg.pulse_gates(gates, 100, 1e-3)
        """
        sequences = dict()
        sequences.update(self.__make_markers(period))
        for gate_name, rel_amplitude in gates.items():
            amplitude = rel_amplitude * sweep_range
            sequences[gate_name] = Sequencer.make_square_wave(amplitude, period)
        sweep_data = self.sequence_gates(sequences, do_upload)
        sweep_data.update({'sweeprange': sweep_range, 'period': period})
        if VirtualAwg.__digitizer_name in self._settings.awg_map:
            sweep_data.update({'markerdelay': self.digitizer_marker_delay()})
        return sweep_data

    def sweep_gates(self, gates, sweep_range, period, width=0.95, do_upload=True):
        """ Supplies a sawtooth wave to the given gates and returns the settings required
            for processing and constucting the readout times for the digitizer.

        Arguments:
            gates (dict): Contains the gate name keys with relative amplitude values.
            sweep_range (float): The overall amplitude of the sawtooth waves in millivolt.
            period (float): The period of the pulse waves in seconds.
            width (float): Width of the rising sawtooth ramp as a proportion of the total cycle.
                           Needs a value between 0 and 1. The value 1 producing a rising ramp,
                           while 0 produces a falling ramp.
            do_upload (bool, Optional): Does not upload the waves to the AWG's when set to False.

        Returns:
            A dictionary with the properties of the pulse waves; the original sawtooth sequence,
            the sweep ranges and the marker properties and period of the sawtooth waves.

        Example:
            >>> sec_period = 1e-6
            >>> mV_sweep_range = 100
            >>> gates = {'P4': 1, 'P7': 0.1}
            >>> sweep_data = virtualawg.sweep_gates(gates, 100, 1e-3)
        """
        sequences = dict()
        sequences.update(self.__make_markers(period))
        for gate_name, rel_amplitude in gates.items():
            amplitude = rel_amplitude * sweep_range
            sequences[gate_name] = Sequencer.make_sawtooth_wave(amplitude, period, width)
        sweep_data = self.sequence_gates(sequences, do_upload)
        sweep_data.update({'sweeprange': sweep_range, 'period': period, 'width': width})
        if VirtualAwg.__digitizer_name in self._settings.awg_map:
            sweep_data.update({'markerdelay': self.digitizer_marker_delay()})
        return sweep_data

    def sweep_gates_2d(self, gates, sweep_ranges, period, resolution, width=0.95, do_upload=True):
        """ Supplies sawtooth signals to a linear combination of gates, which effectively does a 2D scan.

        Arguments:
            gates (list(dict)): A list containing two dicionaries with both the the gate name keys
                                and relative amplitude values.
            sweep_ranges (list): A list two overall amplitude of the sawtooth waves in millivolt in
                                 the x- and y-direction.
            period (float): The period of the sawtooth signals in seconds.
            resolution (list): Two integer values with the number of sawtooth signal (pixels) in the
                               x- and y-direction.
            width (float): Width of the rising sawtooth ramp as a proportion of the total cycle.
                           Needs a value between 0 and 1. The value 1 producing a rising ramp,
                           while 0 produces a falling ramp.
            do_upload (bool, Optional): Does not upload the waves to the AWG's when set to False.

        Returns:
            A dictionary with the properties of the sawtooth signals; the original sawtooth sequence,
            the sweep ranges, the marker properties and period of the sawtooth signals.

        Example:
            >>> sec_period = 1e-6
            >>> resolution = [10, 10]
            >>> mV_sweep_ranges = [100, 100]
            >>> gates = [{'P4': 1}, {'P7': 0.1}]
            >>> sweep_data = virtualawg.sweep_gates_2d(gates, mV_sweep_ranges, period, resolution)
        """
        sequences = dict()

        period_x = resolution[0] * period
        sequences.update(self.__make_markers(period_x, repetitions=resolution[1]))
        for gate_name_x, rel_amplitude_x in gates[0].items():
            amplitude_x = rel_amplitude_x * sweep_ranges[0]
            sequences[gate_name_x] = Sequencer.make_sawtooth_wave(amplitude_x, period_x, width, repetitions=resolution[1])

        period_y = resolution[0] * resolution[1] * period
        for gate_name_y, rel_amplitude_y in gates[1].items():
            amplitude_y = rel_amplitude_y * sweep_ranges[1]
            sequences[gate_name_y] = Sequencer.make_sawtooth_wave(amplitude_y, period_y, width)

        sweep_data = self.sequence_gates(sequences, do_upload)
        sweep_data.update({
            'sweeprange_horz': sweep_ranges[0],
            'sweeprange_vert': sweep_ranges[1],
            'width_horz': width,
            'width_vert': width,
            'resolution': resolution,
            'period': period_y, 'period_horz': period_x,
            'samplerate': self.awgs[0].retrieve_setting('sampling_rate'),
            'markerdelay': self.digitizer_marker_delay()
        })
        return sweep_data

    def pulse_gates_2d(self, gates, sweep_ranges, period, resolution, do_upload=True):
        """ Supplies square signals to a linear combination of gates, which effectively does a 2D scan.

        Arguments:
            gates (list(dict)): A list containing two dicionaries with both the the gate name keys
                                and relative amplitude values.
            sweep_ranges (list): A list two overall amplitude of the square signal in millivolt in
                                 the x- and y-direction.
            period (float): The period of the square signals in seconds.
            resolution (list): Two integer values with the number of square signal (pixels) in the
                               x- and y-direction.
            width (float): Width of the rising square ramp as a proportion of the total cycle.
                           Needs a value between 0 and 1. The value 1 producing a rising ramp,
                           while 0 produces a falling ramp.
            do_upload (bool, Optional): Does not upload the waves to the AWG's when set to False.

        Returns:
            A dictionary with the properties of the square signals; the original square sequence,
            the sweep ranges, the marker properties and period of the square signals.

        Example:
            >>> sec_period = 1e-6
            >>> resolution = [10, 10]
            >>> mV_sweep_ranges = [100, 100]
            >>> gates = [{'P4': 1}, {'P7': 0.1}]
            >>> sweep_data = virtualawg.pulse_gates_2d(gates, mV_sweep_ranges, period, resolution)
        """
        sequences = dict()
        sequences.update(self.__make_markers(period))

        period_x = resolution[0] * period
        for gate_name_x, rel_amplitude_x in gates[0].items():
            amplitude_x = rel_amplitude_x * sweep_ranges[0]
            sequences[gate_name_x] = Sequencer.make_square_wave(amplitude_x, period_x)

        period_y = resolution[0] * resolution[1] * period
        for gate_name_y, rel_amplitude_y in gates[1].items():
            amplitude_y = rel_amplitude_y * sweep_ranges[1]
            sequences[gate_name_y] = Sequencer.make_square_wave(amplitude_y, period_y)

        sweep_data = self.sequence_gates(sequences, do_upload)
        sweep_data.update({'sweeprange_horz': sweep_ranges[0],
                           'sweeprange_vert': sweep_ranges[1],
                           'resolution': resolution,
                           'period': period_x, 'period_vert': period_y,
                           'samplerate': self.awgs[0].retrieve_setting('channel_sampling_rate'),
                           'markerdelay': self.awg_marker_delay()})
        return sweep_data

    def sequence_gates(self, sequences, do_upload=True):
        """ The base function for uploading sequences to the AWG's. The sequences must be
            constructed using the qtt.instument_drivers.virtualAwg.sequencer.Sequencer class.

        Arguments:
            sequences (dict): A dictionary with names as keys and sequences as values.
            do_upload (bool, Optional): Does not upload the waves to the AWG's when set to False.

        Example:
            >>> from qtt.instrument_drivers.virtualAwg.sequencer import Sequencer.
            >>> amplitude_in_volt = 1.5
            >>> period_in_seconds = 1e-6
            >>> sawtooth_signal = Sequencer.make_sawtooth_wave(amplitude_in_volt, period_in_seconds)
            >>> virual_awg.sequence_gates(sawtooth_signal)
        """
        upload_data = []
        if do_upload:
            _ = [awg.delete_waveforms() for awg in self.awgs]
        for number in self.__awg_range:
            sequence_channels = list()
            sequence_names = list()
            sequence_items = list()
            vpp_amplitude = self.awgs[number].retrieve_gain()
            sampling_rate = self.awgs[number].retrieve_sampling_rate()
            for gate_name, sequence in sequences.items():
                (awg_number, channel_number, *marker_number) = self._settings.awg_map[gate_name]
                if awg_number != number:
                    continue

                sequence_data = Sequencer.get_data(sequence, sampling_rate)
                if not marker_number:
                    awg_to_plunger = self._settings.parameters['awg_to_{}'.format(gate_name)].get()
                    scaling_ratio = 2 * VirtualAwg.__volt_to_millivolt * awg_to_plunger / vpp_amplitude
                    sequence_data *= scaling_ratio

                sequence_names.append('{}_{}'.format(gate_name, sequence['name']))
                sequence_channels.append((channel_number, *marker_number))
                sequence_items.append(sequence_data)

            upload_data.append((sequence_names, sequence_channels, sequence_items))
            if do_upload and sequence_items:
                self.awgs[number].upload_waveforms(sequence_names, sequence_channels, sequence_items)
        return {'gate_comb': sequences, 'upload_data': upload_data}


# UNITTESTS #

def test_init_HasNoErrors():
    from unittest.mock import Mock
    awg_driver = Mock()
    type(awg_driver).__name__ = 'Tektronix_AWG5014'
    awgs = [awg_driver]

    class QuantumDeviceSettings(Instrument):

        def __init__(self):
            super().__init__('settings')
            self.awg_map = {
                'P1': (0, 1),
                'P2': (0, 2),
                'dig_mk': (0, 1, 1)
            }

    settings = QuantumDeviceSettings()
    virtual_awg = VirtualAwg(awgs, settings)
    assert awg_driver == virtual_awg.awgs[0].fetch_awg
