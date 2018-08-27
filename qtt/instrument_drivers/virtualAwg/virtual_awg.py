import logging
import numpy as np

from qcodes import Instrument
from qcodes.utils.validators import Numbers
from qtt.instrument_drivers.virtualAwg.sequencer import Sequencer
from qtt.instrument_drivers.virtualAwg.awgs.simulated_awg import Simulated_AWG
from qtt.instrument_drivers.virtualAwg.awgs.Tektronix5014C import Tektronix5014C_AWG
try:
    from qtt.instrument_drivers.virtualAwg.awgs.KeysightM3202A import KeysightM3202A_AWG
except:
    logging.debug('could not load KeysightM3202A driver')
    KeysightM3202A_AWG=None
from qtt.instrument_drivers.virtualAwg.awgs.ZurichInstrumentsHDAWG8 import ZurichInstruments_HDAWG8


class VirtualAwgError(Exception):
    """Exception for a specific error related to the virtual AWG."""


class VirtualAwg(Instrument):

    __volt_to_millivolt = 1e-3

    def __init__(self, awgs, gate_map, hardware, name='virtual_awg', logger=logging, **kwargs):
        super().__init__(name, **kwargs)
        self._gate_map = gate_map
        self.__hardware = hardware
        self.__set_hardware(awgs)
        self.__set_parameters()
        self._logger = logger

    def _get_virtual_info(self):
        """ Return data needed for snapshot of instrument """
        return {'gate_map': self._gate_map, 'awgs': [type(awg).__name__ for awg in self.awgs]}

    def __set_hardware(self, awgs):
        self.awgs = list()
        for awg in awgs:
            if type(awg).__name__ == 'Tektronix_AWG5014':
                self.awgs.append(Tektronix5014C_AWG(awg))
            elif type(awg).__name__ == 'Keysight_M3201A':
                self.awgs.append(KeysightM3202A_AWG(awg))
            elif type(awg).__name__ == 'ZI_HDAWG8':
                self.awgs.append(ZurichInstruments_HDAWG8(awg))
            elif type(awg).__name__ == 'Mock':
                self.awgs.append(Simulated_AWG(awg))
            else:
                raise VirtualAwgError('Unusable device added!')
        self.__awg_range = range(0, len(self.awgs))
        self.__awg_count = len(self.awgs)

    def __set_parameters(self):
        self.add_parameter('awg_marker_delay', initial_value=0, vals=Numbers(0, 1), set_cmd=None)
        self.add_parameter('awg_marker_uptime', initial_value=0.2, vals=Numbers(0, 1), set_cmd=None)
        self.add_parameter('digitizer_marker_delay', initial_value=0.2, vals=Numbers(0, 1), set_cmd=None)
        self.add_parameter('digitizer_marker_uptime', initial_value=0.2, vals=Numbers(0, 1), set_cmd=None)

    def run(self):
        [awg.run() for awg in self.awgs]

    def stop(self):
        [awg.stop() for awg in self.awgs]

    def reset(self):
        [awg.reset() for awg in self.awgs]

    def enable_outputs(self, gate_names):
        gate_names.extend(['m4i_mk', 'awg_mk'])
        for name in gate_names:
            (awg_number, channel_number, *_) = self._gate_map[name]
            self.awgs[awg_number].enable_outputs([channel_number])

    def disable_outputs(self, gate_names):
        gate_names.extend(['m4i_mk', 'awg_mk'])
        for name in gate_names:
            (awg_number, channel_number, *_) = self._gate_map[name]
            self.awgs[awg_number].disable_outputs([channel_number])

    def update_setting(self, awg_number, setting, value):
        if awg_number not in self.__awg_range:
            raise VirtualAwgError('Invalid AWG number {}!'.format(awg_number))
        self.awgs[awg_number].update_setting(setting, value)

    def are_awg_gates(self, gate_names):
        if gate_names is None:
            return False
        if isinstance(gate_names, list):
            return np.all([self.are_awg_gates(g) for g in gate_names])
        return True if gate_names in self._gate_map else False

    def __make_markers(self, period):
        digitizer_marker = Sequencer.make_marker(period, self.digitizer_marker_uptime(), self.digitizer_marker_delay())
        awg_marker = Sequencer.make_marker(period, self.awg_marker_uptime(), self.awg_marker_delay())
        return dict({'m4i_mk': digitizer_marker, 'awg_mk': awg_marker})

    def pulse_gates(self, gates, sweep_range, period, do_upload=True):
        sequences = dict()
        sequences.update(self.__make_markers(period))
        for gate_name, rel_amplitude in gates.items():
            amplitude = rel_amplitude * sweep_range
            sequences[gate_name] = Sequencer.make_square_wave(amplitude, period)
        sweep_data = self.sequence_gates(sequences, do_upload)
        return sweep_data.update({'sweeprange': sweep_range, 'period': period,
                                  'markerdelay': self.digitizer_marker_delay()})

    def sweep_gates(self, gates, sweep_range, period, width=0.95, do_upload=True):
        """ Sweep a set of gates with a sawtooth waveform.

        Example:
            >>> sweep_data = virtualawg.sweep_gates({'P4': 1, 'P7': 0.1}, 100, 1e-3)
        """
        sequences = dict()
        sequences.update(self.__make_markers(period))
        for gate_name, rel_amplitude in gates.items():
            amplitude = rel_amplitude * sweep_range
            sequences[gate_name] = Sequencer.make_sawtooth_wave(amplitude, period, width)
        sweep_data = self.sequence_gates(sequences, do_upload)
        sweep_data.update({'sweeprange': sweep_range, 'period': period, 'width': width,
                           'markerdelay': self.digitizer_marker_delay()})
        return sweep_data

    def sweep_gates_2d(self, gates, sweep_ranges, period, resolution, width=0.95, do_upload=True):
        sequences = dict()
        sequences.update(self.__make_markers(period))

        period_x = resolution[0] * period
        for gate_name_x, rel_amplitude_x in gates[0].items():
            amplitude_x = rel_amplitude_x * sweep_ranges[0]
            sequences[gate_name_x] = Sequencer.make_sawtooth_wave(amplitude_x, period_x, width)

        period_y = resolution[0] * resolution[1] * period
        for gate_name_y, rel_amplitude_y in gates[1].items():
            amplitude_y = rel_amplitude_y * sweep_ranges[1]
            sequences[gate_name_y] = Sequencer.make_sawtooth_wave(amplitude_y, period_y, width)

        sweep_data = self.sequence_gates(sequences, do_upload)
        sweep_data.update({'sweeprange_horz': sweep_ranges[0],
                           'sweeprange_vert': sweep_ranges[1],
                           'width_horz': 0.95,
                           'width_vert': 0.95,
                           'resolution': resolution,
                           'period': period_y, 'period_horz': period_x,
                           'samplerate': self.awgs[0].retrieve_setting('sampling_rate'),
                           'markerdelay': self.awg_marker_delay()})
        return sweep_data

    def pulse_gates_2d(self, gates, sweep_ranges, period, resolution, do_upload=True):
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
        return sweep_data.update({'sweeprange_horz': sweep_ranges[0],
                                  'sweeprange_vert': sweep_ranges[1],
                                  'resolution': resolution,
                                  'period': period_x, 'period_vert': period_y,
                                  'samplerate': self.awgs[0].retrieve_setting('channel_sampling_rate'),
                                  'markerdelay': self.awg_marker_delay()})

    def sequence_gates(self, gate_comb, do_upload=True):
        if do_upload:
            [awg.delete_waveforms() for awg in self.awgs]
        for number in self.__awg_range:
            sequence_channels = list()
            sequence_names = list()
            sequence_items = list()
            vpp_amplitude = self.awgs[number].retrieve_gain()
            sampling_rate = self.awgs[number].retrieve_sampling_rate()
            for gate_name, sequence in gate_comb.items():
                (awg_number, channel_number, *marker_number) = self._gate_map[gate_name]
                if awg_number != number:
                    continue
                awg_to_plunger = self.__hardware.parameters['awg_to_{}'.format(gate_name)].get()
                scaling_ratio = 2 * VirtualAwg.__volt_to_millivolt / awg_to_plunger / vpp_amplitude

                sample_data = Sequencer.get_data(sequence, sampling_rate)
                sequence_data = sample_data if marker_number else sample_data * scaling_ratio

                sequence_names.append('{}_{}'.format(gate_name, sequence['name']))
                sequence_channels.append((channel_number, *marker_number))
                sequence_items.append(sequence_data)
            if do_upload:
                self.awgs[number].upload_waveforms(sequence_names, sequence_channels, sequence_items)
        return {'gate_comb': gate_comb}


# UNITTESTS #

def test_init_HasNoErrors():
    from unittest.mock import Mock
    awg_driver = Mock(name='simulated_awg')
    awgs = [awg_driver]
    gate_map = {'P1': (0, 1), 'P2': (0, 2), 'digitizer_marker': (0, 1, 1)}
    virtual_awg = VirtualAwg(awgs, gate_map, hardware=None)
    assert awg_driver == virtual_awg.awgs[0].fetch_awg
