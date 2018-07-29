import logging
import numpy as np

from functools import partial
from qcodes import Instrument
from qcodes.utils.validators import Numbers
from qtt.instrument_drivers.virtualAwg.sequencer import Sequencer
from qtt.instrument_drivers.virtualAwg.awgs.simulated_awg import Simulated_AWG
from qtt.instrument_drivers.virtualAwg.awgs.Tektronix5014C import Tektronix5014C_AWG
from qtt.instrument_drivers.virtualAwg.awgs.KeysightM3202A import KeysightM3202A_AWG
from qtt.instrument_drivers.virtualAwg.awgs.ZurichInstrumentsHDAWG8 import ZurichInstruments_HDAWG8


class VirtualAwgError(Exception):
    """Exception for a specific error related to the virtual AWG."""


class VirtualAwg(Instrument):

    __volt_to_millivolt = 1e-3

    def __init__(self, awgs, gate_map, hardware, name='virtual_awg', logger=logging, **kwargs):
        super().__init__(name, **kwargs)
        self.__gate_map = gate_map
        self.__hardware = hardware
        self.__set_hardware(awgs)
        self.__logger = logger

    def _get_virtual_info(self):
        """ Return data needed for snapshot of instrument """
        return {'gate_map': self.__gate_map, 'awgs': [str(a) for a in self.awgs]}

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
        self.add_parameter('marker_delay', initial_value=0, vals=Numbers(0, 1))
        self.add_parameter('marker_uptime', initial_value=0.2, vals=Numbers(0, 1))

    def run(self):
        [awg.run() for awg in self.awgs]

    def stop(self):
        [awg.stop() for awg in self.awgs]

    def reset(self):
        [awg.reset() for awg in self.awgs]

    def enable_outputs(self, gate_names):
        for name in gate_names:
            (awg_nr, channel_nr) = self.__gate_map[name]
            self.awgs[awg_nr].enable_outputs([channel_nr])

    def disable_outputs(self, gate_names):
        for name in gate_names:
            (awg_nr, channel_nr) = self.__gate_map[name]
            self.awgs[awg_nr].disable_outputs([channel_nr])

    def update_setting(self, awg_nr, setting, value):
        if awg_nr not in self.__awg_range:
            raise VirtualAwgError('Invalid AWG nr {}!'.format(awg_nr))
        self.awgs[awg_nr].update_setting(setting, value)

    def are_awg_gates(self, gate_names):
        if gate_names is None:
            return False
        if isinstance(gate_names, list):
            return np.all([self.are_awg_gates(g) for g in gate_names])
        return True if gate_names in self.__gate_map else False

    @staticmethod
    def __update_sequence(sequences, gate_comb, sequence_function):
        max_vpp = 0.0
        for gate_name, amplitude in gate_comb:
            volt_peak_to_peak = amplitude * VirtualAwg.__volt_to_millivolt
            if abs(volt_peak_to_peak) > max_vpp:
                max_vpp = abs(volt_peak_to_peak)
            sequence_data = sequence_function(volt_peak_to_peak)
            sequences[gate_name] = sequence_data
        return max_vpp

    def __initialize_gates(self, gate_comb, period, sequence_function, do_upload=True):
        digitizer_marker = Sequencer.make_marker(period, self.marker_uptime, self.marker_offset)
        awg_marker = Sequencer.make_marker(period, self.marker_uptime, self.marker_offset)
        sequences = {'m4i_mk': digitizer_marker, 'awg_mk': awg_marker}
        function = partial(sequence_function, period=period)
        max_vpp = VirtualAwg.__update_sequence(sequences, gate_comb, function)
        sweep_data = self.sequence_gates(sequences, do_upload)
        return sweep_data.update({'sweeprange': max_vpp,
                                  'period': period, 'markerdelay': self.marker_offset})

    def pulse_gates(self, gate_comb, period, do_upload=True):
        sequence_function = partial(Sequencer.make_square_wave)
        return self.__initialize_gates(gate_comb, period, sequence_function, do_upload)

    def sweep_gates(self, gate_comb, period, width=0.95, do_upload=True):
        """ Sweep a set of gates with a sawtooth waveform.

        Example:
            >>> sweep_data = virtualawg.sweep_gates({'P4': 1e-3, 'P7':-1e-3}, 1e-3)
        """
        sequence_function = partial(Sequencer.make_sawtooth_wave, width=width)
        sweep_data = self.__initialize_gates(gate_comb, period, sequence_function, do_upload)
        sweep_data.update({'width': width})
        return sweep_data

    def __initialize_gates_2D(self, gate_combs, period, resolution, sequence_function, do_upload=True):
        digitizer_marker = Sequencer.make_marker(period, self.marker_uptime, self.marker_offset)
        awg_marker = Sequencer.make_marker(period, self.marker_uptime, self.marker_offset)
        sequences = {'m4i_mk': digitizer_marker, 'awg_mk': awg_marker}

        gate_comb_x = gate_combs[0]
        period_x = resolution[0] * period
        function_x = partial(sequence_function, period=period_x)
        max_vpp_x = VirtualAwg.__update_sequence(self, gate_comb_x, function_x)

        gate_comb_y = gate_combs[1]
        period_y = resolution[1] * resolution[1] * period
        function_y = partial(sequence_function, period=period_y)
        max_vpp_y = VirtualAwg.__update_sequence(self, gate_comb_y, function_y)

        sweep_data = self.sequence_gates(sequences, do_upload)
        return sweep_data.update({'sweeprange_horz': max_vpp_x, 'sweeprange_vert': max_vpp_y,
                                  'resolution': resolution, 'period': period_x, 'period_vert': period_y,
                                  'samplerate': self.awgs[0].get_sampling_rate(),
                                  'markerdelay': self.marker_offset})

    def sweep_gates_2D(self, gate_comb_x, gate_comb_y, period, resolution, width=0.95, do_upload=True):
        gate_combs = (gate_comb_x, gate_comb_y)
        function = partial(Sequencer.make_sawtooth_wave, width=width)
        sweep_data = self.__initialize_gates_2D(gate_combs, period, resolution, function, do_upload)
        sweep_data.update({'width_horz': width, 'width_vert': width})
        return sweep_data

    def pulse_gates_2D(self, gate_comb_x, gate_comb_y, period, resolution, do_upload=True):
        gate_combs = (gate_comb_x, gate_comb_y)
        function = partial(Sequencer.make_square_wave)
        return self.__initialize_gates_2D(gate_combs, period, resolution, function, do_upload)

    def sequence_gates(self, gate_comb, do_upload=True):
        if do_upload:
            [awg.delete_waveforms() for awg in self.awgs]
        for nr in self.__awg_range:
            sequence_channels = list()
            sequence_names = list()
            sequence_items = list()
            vpp_amplitude = self.awgs[nr].retrieve_gain()
            sampling_rate = self.awgs[nr].retrieve_sampling_rate()
            for gate_name, sequence in gate_comb.items():
                (awg_nr, channel_nr, *marker_nr) = self.__gate_map[gate_name]
                if awg_nr != nr:
                    continue
                awg_to_plunger = self.__hardware.parameters['awg_to_{}'.format(gate_name)].get()
                scaling_ratio = 2.0 / VirtualAwg.__volt_to_millivolt / awg_to_plunger / vpp_amplitude

                sample_data = Sequencer.get_data(sequence, sampling_rate)
                sequence_items.append(sequence)
                sequence_data = sample_data if marker_nr else sample_data * scaling_ratio

                sequence_names.append('{}_{}'.format(gate_name, sequence['name']))
                sequence_channels.append((channel_nr, *marker_nr))
                sequence_items.append(sequence_data)
        if do_upload:
            self.awgs[nr].upload_waveforms(sequence_names, sequence_channels, sequence_items)
        return {'gate_comb': gate_comb}


# UNITTESTS #

def test_init_HasNoErrors():
    from unittest.mock import Mock
    awg_driver = Mock(name='simulated_awg')
    awgs = [awg_driver]
    gate_map = {'P1': (0, 1), 'P2': (0, 2), 'digitizer_marker': (0, 1, 1)}
    virtual_awg = VirtualAwg(awgs, gate_map, hardware=None)
    assert(awg_driver == virtual_awg.awgs[0].fetch_awg)
