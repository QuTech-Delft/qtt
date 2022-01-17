import logging
from typing import List

import numpy as np
from qcodes import Instrument

from qtt.instrument_drivers.virtualAwg.awgs.common import AwgCommon
from qtt.instrument_drivers.virtualAwg.awgs.KeysightM3202A import KeysightM3202A_AWG
from qtt.instrument_drivers.virtualAwg.awgs.Tektronix5014C import Tektronix5014C_AWG
from qtt.instrument_drivers.virtualAwg.awgs.ZurichInstrumentsHDAWG8 import ZurichInstrumentsHDAWG8
from qtt.instrument_drivers.virtualAwg.sequencer import Sequencer
from qtt.instrument_drivers.virtualAwg.settings import SettingsInstrument


class VirtualAwgError(Exception):
    """ Exception for a specific error related to the virtual AWG."""


class VirtualAwg(Instrument):
    """ The virtual AWG is an abstraction layer to produce pulse driven state manipulation of physical qubits.

    The class aims for hardware independent control, where only common arbitrary waveform generator (AWG)
    functionality is used. A translation between the AWG channels and the connected quantum gates provide
    the user control of the AWG's in terms of gate names and waveform sequences only. The virtual AWG is
    used for fast change of the DC landscape by changing the voltage levels of the gates. No microwave
    control is involved, meaning not related to the spin degrees of freedom of the qubits.


    Attributes:
        enable_debug (bool): If Tre that store intermediate results in debugging variables

    """

    __digitizer_name = 'm4i_mk'
    __awg_slave_name = 'awg_mk'
    __volt_to_millivolt = 1e3

    def __init__(self, awgs=None, settings=None, name='virtual_awg', logger=logging, **kwargs):
        """ Creates and initializes an virtual AWG object and sets the relation between the quantum gates,
            markers and the AWG channels. The default settings (marker delays) are constructed at startup.

        Arguments:
            awgs (list): A list with AWG instances. Currently the following AWG's are supported:
                         Tektronix5014C, KeysightM3202A and the ZurichInstrumentsHDAWG8.
            settings (Instrument): A class containing the settings of the quantum device, which are the
                                  awg_map (specifies the relation between the quantum gates, marker outputs
                                  and AWG channels) and the awg_to_gate [mV/V] (specifies how many millivolt
                                  on a sample is induced by an output in Volt from the AWG).
        """
        super().__init__(name, **kwargs)
        self._settings = settings
        self._logger = logger
        if awgs is None:
            awgs = []
        self._awgs = []
        self.settings = settings
        self._instruments = []
        self.add_instruments(awgs)
        self._latest_sequence_data = {}
        self.enable_debug = False

        self.add_parameter('settings_snapshot', get_cmd=self.settings.snapshot, label='Settings snapshot')
        self.settings_snapshot()

    def _get_virtual_info(self):
        """ Returns the data needed for snapshot of instrument."""
        return {'awg_map': self._settings.awg_map, 'awgs': [type(awg).__name__ for awg in self.awgs]}

    def __set_hardware(self, awgs):
        """ Sets the virtual AWG backends using the QCoDeS driver. Currently the following AWG are supported:
            the Tektronix AWG5014, the Keysight M3201A and the Zurich Instruments HDAWG8.

        Arguments:
            awgs (list): A list with the QCoDeS driver instances.
        """
        self._awgs = []
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

        self.parameters.pop('awg_marker_delay', None)
        self.parameters.pop('awg_marker_uptime', None)
        self.parameters.pop('digitizer_marker_delay', None)
        self.parameters.pop('digitizer_marker_uptime', None)

        if VirtualAwg.__awg_slave_name in self._settings.awg_map:
            self.add_parameter('awg_marker_delay', initial_value=0, set_cmd=None)
            self.add_parameter('awg_marker_uptime', initial_value=1e-8, set_cmd=None)
        if VirtualAwg.__digitizer_name in self._settings.awg_map:
            self.add_parameter('digitizer_marker_delay', initial_value=0, set_cmd=None, unit='s')
            self.add_parameter('digitizer_marker_uptime', initial_value=1e-8, set_cmd=None, unit='s')

    @property
    def settings(self) -> SettingsInstrument:
        """ The device's settings. """
        return self._settings

    @settings.setter
    def settings(self, value: SettingsInstrument) -> None:
        """ Change the device's settings and update its parameters. """
        self._settings = value
        if value is not None:
            self.__set_parameters()

    @property
    def awgs(self) -> List[AwgCommon]:
        """ The device's awgs. """
        return self._awgs

    @property
    def instruments(self) -> List[Instrument]:
        """ The device's instruments. """
        return self._instruments

    @instruments.setter
    def instruments(self, value: List[Instrument]) -> None:
        """
        Updates the devices instruments

        Args:
            value: The list of instruments to update
        """
        self._instruments = value
        self.__set_hardware(value)

    def add_instruments(self, instruments: List[Instrument]):
        """
        Adds a list of instruments and updates its hardware parameters.

        Args:
            instruments: The list of instruments to add
        """
        for instrument in instruments:
            if instrument not in self._instruments:
                self._instruments.append(instrument)

        self.__set_hardware(self._instruments)

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
            gate_names (list[str]): The names of the gates which needs to be enabled.
        """
        if VirtualAwg.__digitizer_name in self._settings.awg_map:
            gate_names.extend([VirtualAwg.__digitizer_name])
        if VirtualAwg.__awg_slave_name in self._settings.awg_map:
            gate_names.extend([VirtualAwg.__awg_slave_name])
        for name in gate_names:
            (awg_number, channel_number, *_) = self._settings.awg_map[name]
            self.awgs[awg_number].enable_outputs([channel_number])

    def disable_outputs(self, gate_names=None):
        """ Sets the given gates output to disabled. The gate map translates the given gate
            names to the correct AWG and channels. The digitizer and awg marker channels
            are automatically disabled if the channels are provided by the setttings awg_map.
            A start command is required to enable the outputs.

        Arguments:
            gate_names (list or None) The names of the gates which needs to be disabled. If None, then disable all gates
        """
        if gate_names is None:
            for awg in self.awgs:
                awg.disable_outputs()
        else:
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
            raise VirtualAwgError(f'Invalid AWG number {awg_number}!')
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
        if (VirtualAwg.__digitizer_name in gate_names) or (VirtualAwg.__awg_slave_name in gate_names):
            return False
        return True if gate_names in self._settings.awg_map else False

    def make_markers(self, period, repetitions=1):
        """ Constructs the markers for triggering the digitizer readout and the slave AWG
            start sequence. The sequence length equals the period x repetitions.

        Arguments:
            period (float): The period of the markers in seconds.
            repetitions (int): The number of markers in the sequence.
        """
        marker_properties = {}
        uptime = self.digitizer_marker_uptime()
        delay = self.digitizer_marker_delay()

        if VirtualAwg.__digitizer_name in self._settings.awg_map:
            digitizer_marker = Sequencer.make_marker(period, uptime, delay, repetitions)
            marker_properties[VirtualAwg.__digitizer_name] = [digitizer_marker]

        if VirtualAwg.__awg_slave_name in self._settings.awg_map:
            awg_marker = Sequencer.make_marker(period, uptime, delay, repetitions)
            marker_properties[VirtualAwg.__awg_slave_name] = [awg_marker]

        return marker_properties

    def update_digitizer_marker_settings(self, uptime, delay):
        """ Updates the marker settings of the AWG to trigger the digitizer. Note that the
            uptime and delay time in seconds must not be bigger then the period of the
            uploaded waveform.

        Arguments:
            uptime (float): The marker up period in seconds.
            delay (float): The marker delay in seconds.
        """
        if VirtualAwg.__digitizer_name not in self._settings.awg_map:
            raise ValueError('Digitizer marker not present in settings awg map!')

        self.digitizer_marker_uptime(uptime)
        self.digitizer_marker_delay(delay)

    def update_slave_awg_marker_settings(self, uptime, delay):
        """ Updates the marker settings of the AWG to trigger the other AWG's. Note that the
            uptime and delay time in seconds must not be bigger then the period of the
            uploaded waveform.

        Arguments:
            uptime (float): The marker up period in seconds.
            delay (float): The marker delay in seconds.
        """
        if VirtualAwg.__awg_slave_name not in self._settings.awg_map:
            raise ValueError('Slave AWG marker not present in settings awg map!')

        self.awg_marker_uptime(uptime)
        self.awg_marker_delay(delay)

    def pulse_gates(self, gate_voltages, waiting_times, repetitions=1, do_upload=True):
        """ Supplies custom sequences to the gates. The supplied list of voltage setpoints with
            waiting times are converted into sequences for each gate and upload to the AWG.

        Arguments:
            gate_voltages (dict): Each gate name key contains a an array with millivolt
                                  setpoint level to be converted into a sequence.
            waiting_times (list[float]): The duration in seconds of each pulse in the sequence.
            repetitions (int): The number of times to repeat the sequence.
            do_upload (bool, Optional): Does not upload the waves to the AWG's when set to False.

        Returns:
            A dictionary with the properties of the pulse waves; the original pulse sequence,
            the sweep ranges and the marker properties and period of the pulse waves.

        Example:
            >> gates_voltages = {'P4': [50, 0, -50], 'P7': [-25, 0, 25]}
            >> waiting_times = [1e-4, 1e-4, 1e-4]
            >> pulse_data = virtual_awg.pulse_gates(gate_voltages, waiting_times)
        """
        sequences = {}
        period = sum(waiting_times)
        sequences.update(self.make_markers(period, repetitions))

        for gate_name, amplitudes in gate_voltages.items():
            pulse_wave = Sequencer.make_pulse_table(amplitudes, waiting_times, repetitions, gate_name)
            sequences.setdefault(gate_name, []).append(pulse_wave)

        sweep_data = self.sequence_gates(sequences, do_upload)
        sweep_data.update({'period': period,
                           'start_zero': True,
                           'width': 1.0})

        if VirtualAwg.__digitizer_name in self._settings.awg_map:
            sweep_data.update({'markerdelay': self.digitizer_marker_delay()})

        return sweep_data

    def sweep_gates(self, gates, sweep_range, period, width=0.9375, do_upload=True, zero_padding=0):
        """ Supplies a sawtooth wave to the given gates and returns the settings required
            for processing and constructing the readout times for the digitizer.

        Arguments:
            gates (dict): Contains the gate name keys with relative amplitude values.
            sweep_range (float): The peak-to-peak amplitude of the sawtooth waves in millivolt.
            period (float): The period of the pulse waves in seconds.
            width (float): Width of the rising sawtooth ramp as a proportion of the total cycle.
                           Needs a value between 0 and 1. The value 1 producing a rising ramp,
                           while 0 produces a falling ramp.
            do_upload (bool, Optional): Does not upload the waves to the AWG's when set to False.
            zero_padding (float): Amount of zero padding to add (in seconds)

        Returns:
            A dictionary with the properties of the pulse waves; the original sawtooth sequence,
            the sweep ranges and the marker properties and period of the sawtooth waves.

        Example:
            >> sec_period = 1e-6
            >> mV_sweep_range = 100
            >> gates = {'P4': 1, 'P7': 0.1}
            >> sweep_data = virtual_awg.sweep_gates(gates, 100, 1e-3)
        """
        sequences = {}
        sequences.update(self.make_markers(period + zero_padding))

        for gate_name, rel_amplitude in gates.items():
            amplitude = rel_amplitude * sweep_range
            sweep_wave = Sequencer.make_sawtooth_wave(amplitude, period, width, zero_padding=zero_padding)
            sequences.setdefault(gate_name, []).append(sweep_wave)

        sweep_data = self.sequence_gates(sequences, do_upload)
        sweep_data.update({'sweeprange': sweep_range,
                           'period': period,
                           'width': width,
                           'start_zero': True,
                           'zero_padding': zero_padding,
                           '_gates': gates})

        if VirtualAwg.__digitizer_name in self._settings.awg_map:
            sweep_data.update({'markerdelay': self.digitizer_marker_delay()})

        return sweep_data

    def sweep_gates_2d(self, gates, sweep_ranges, period, resolution, width=0.9375, do_upload=True):
        """ Supplies sawtooth signals to a linear combination of gates, which effectively does a 2D scan.

        Arguments:
            gates (list[dict]): A list containing two dictionaries with both the the gate name keys
                                and relative amplitude values.
            sweep_ranges (list): A list two overall amplitude of the sawtooth waves in millivolt in
                                 the x- and y-direction.
            period (float): The total period of the sawtooth signals in seconds.
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
            >> sec_period = 1e-6
            >> resolution = [10, 10]
            >> mV_sweep_ranges = [100, 100]
            >> gates = [{'P4': 1}, {'P7': 0.1}]
            >> sweep_data = virtual_awg.sweep_gates_2d(gates, mV_sweep_ranges, period, resolution)
        """
        sequences = {}
        base_period = period / np.prod(resolution)
        sequences.update(self.make_markers(period, repetitions=1))

        period_x = resolution[0] * base_period
        for gate_name_x, rel_amplitude_x in gates[0].items():
            amplitude_x = rel_amplitude_x * sweep_ranges[0]
            sweep_wave_x = Sequencer.make_sawtooth_wave(amplitude_x, period_x, width, resolution[1])
            sequences.setdefault(gate_name_x, []).append(sweep_wave_x)

        period_y = resolution[0] * resolution[1] * base_period
        for gate_name_y, rel_amplitude_y in gates[1].items():
            amplitude_y = rel_amplitude_y * sweep_ranges[1]
            sweep_wave_y = Sequencer.make_sawtooth_wave(amplitude_y, period_y, width)
            sequences.setdefault(gate_name_y, []).append(sweep_wave_y)

        sweep_data = self.sequence_gates(sequences, do_upload)
        sweep_data.update({'sweeprange_horz': sweep_ranges[0],
                           'sweeprange_vert': sweep_ranges[1],
                           'width_horz': width,
                           'width_vert': width,
                           'resolution': resolution,
                           'start_zero': True,
                           'period': period_y, 'period_horz': period_x,
                           'samplerate': self.awgs[0].retrieve_sampling_rate(),
                           'markerdelay': self.digitizer_marker_delay()})
        return sweep_data

    def pulse_gates_2d(self, gates, sweep_ranges, period, resolution, do_upload=True):
        """ Supplies square signals to a linear combination of gates, which effectively does a 2D scan.

        Arguments:
            gates (list[dict]): A list containing two dictionaries with both the the gate name keys
                                and relative amplitude values.
            sweep_ranges (list): A list two overall amplitude of the square signal in millivolt in
                                 the x- and y-direction.
            period (float): The period of the square signals in seconds.
            resolution (list): Two integer values with the number of square signal (pixels) in the
                               x- and y-direction.
            do_upload (bool, Optional): Does not upload the waves to the AWG's when set to False.

        Returns:
            A dictionary with the properties of the square signals; the original square sequence,
            the sweep ranges, the marker properties and period of the square signals.

        Example:
            >> sec_period = 1e-6
            >> resolution = [10, 10]
            >> mV_sweep_ranges = [100, 100]
            >> gates = [{'P4': 1}, {'P7': 0.1}]
            >> sweep_data = virtual_awg.pulse_gates_2d(gates, mV_sweep_ranges, period, resolution)
        """
        sequences = {}
        sequences.update(self.make_markers(period))

        period_x = resolution[0] * period
        for gate_name_x, rel_amplitude_x in gates[0].items():
            amplitude_x = rel_amplitude_x * sweep_ranges[0]
            pulse_wave_x = Sequencer.make_square_wave(amplitude_x, period_x, resolution[1])
            sequences.setdefault(gate_name_x, []).append(pulse_wave_x)

        period_y = resolution[0] * resolution[1] * period
        for gate_name_y, rel_amplitude_y in gates[1].items():
            amplitude_y = rel_amplitude_y * sweep_ranges[1]
            pulse_wave_y = Sequencer.make_square_wave(amplitude_y, period_y)
            sequences.setdefault(gate_name_y, []).append(pulse_wave_y)

        sweep_data = self.sequence_gates(sequences, do_upload)
        sweep_data.update({'sweeprange_horz': sweep_ranges[0],
                           'sweeprange_vert': sweep_ranges[1],
                           'resolution': resolution,
                           'period': period_x,
                           'period_vert': period_y,
                           'samplerate': self.awgs[0].retrieve_setting('channel_sampling_rate'),
                           'markerdelay': self.awg_marker_delay()})
        return sweep_data

    def sequence_gates(self, sequences, do_upload=True):
        """ The base function for uploading sequences to the AWG's. The sequences must be
            constructed using the qtt.instrument_drivers.virtualAwg.sequencer.Sequencer class.

        Arguments:
            sequences (dict): A dictionary with names as keys and sequences as values.
            do_upload (bool, Optional): Does not upload the waves to the AWG's when set to False.

        Example:
            >> from qtt.instrument_drivers.virtualAwg.sequencer import Sequencer.
            >> amplitude = 1.5
            >> period_in_seconds = 1e-6
            >> sawtooth_signal = Sequencer.make_sawtooth_wave(amplitude, period_in_seconds)
            >> virtual_awg.sequence_gates(sawtooth_signal)
        """
        upload_data = []
        settings_data = {}

        if do_upload:
            _ = [awg.delete_waveforms() for awg in self.awgs]

        for number in self.__awg_range:
            sequence_channels = []
            sequence_names = []
            sequence_items = []

            gain_factor = self.awgs[number].retrieve_gain()
            vpp_amplitude = 2 * gain_factor
            sampling_rate = self.awgs[number].retrieve_sampling_rate()
            settings_data[number] = {'vpp_amplitude': vpp_amplitude, 'sampling_rate': sampling_rate}

            for gate_name, sequence in sequences.items():
                (awg_number, channel_number, *marker_number) = self._settings.awg_map[gate_name]
                if awg_number != number:
                    continue

                waveforms = [Sequencer.get_data(waveform, sampling_rate) for waveform in sequence]
                sequence_data = np.sum(waveforms, 0)
                sequence_data = sequence_data[:-1]
                if not marker_number:
                    awg_to_gate = self._settings.parameters[f'awg_to_{gate_name}'].get()
                    scaling_ratio = 1 / (awg_to_gate * gain_factor)
                    settings_data[number][gate_name] = {'scaling_ratio': scaling_ratio}
                    sequence_data *= scaling_ratio

                sequence_name = sequence[0]['name']
                sequence_names.append(f'{gate_name}_{sequence_name}')
                sequence_channels.append((channel_number, *marker_number))
                sequence_items.append(sequence_data)

            upload_data.append((sequence_names, sequence_channels, sequence_items))
            if do_upload and sequence_items:
                self.awgs[number].upload_waveforms(sequence_names, sequence_channels, sequence_items)

        sequence_data = {'gate_comb': sequences, 'upload_data': upload_data, 'settings': settings_data}
        if self.enable_debug:
            self._latest_sequence_data = sequence_data
        return sequence_data
