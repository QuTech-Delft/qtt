import logging
import numpy as np
import qtt.instrument_drivers.virtualawg.awgs
import qtt.instrument_drivers.virtualawg.awgs.Tektronix5014C as Tektronix5014C
import qtt.instrument_drivers.virtualawg.awgs.KeysightM3202A as KeysightM3202A

from qcodes import Instrument
from qtt.instrument_drivers.virtualawg.sequencer import Sequencer


class VirtualAwgError(Exception):
    """Exception for a specific error related to the virtual AWG."""


class VirtualAwg(Instrument):

    __volt_to_millivolt = 1e-3

    def __init__(self, awgs, gate_map, hardware, name='virtual_awg', logger=logging, **kwargs):
        super().__init__(name, **kwargs)
        self.__hardware = hardware
        self.__gate_map = gate_map
        self.__set_hardware(awgs)
        self.__logger = logger

    def __set_hardware(self, awgs):
        self.awgs = list()
        for awg in awgs:
            if type(awg).__name__ == 'Tektronix_AWG5014':
                self.awgs.append(Tektronix5014C.Tektronix5014C_AWG(awg))
            elif type(awg).__name__ == 'Keysight_M3201A':
                self.awgs.append(KeysightM3202A.KeysightM3202A_AWG(awg))
            else:
                raise VirtualAwgError('Unusable device added!')
        self.__awg_range = range(0, len(self.awgs))
        self.__awg_count = len(self.awgs)

    def run_awgs(self):
        [awg.run() for awg in self.awgs]

    def stop_awgs(self):
        [awg.stop() for awg in self.awgs]

    def reset_awgs(self):
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

    def sweep_gates(self, gate_names, amplitudes, period, width=0.95, marker_uptime=0.2, marker_offset=0.0):
        if type(gate_names) == 'str':
            gate_names = [gate_names]
            amplitudes = [amplitudes]
        sequences = list()
        for gate_name, amplitude in zip(gate_names, amplitudes):
            (awg_nr, channel_nr, *marker_nr) = self.__gate_map[gate_name]
            if marker_nr:
                marker = Sequencer.make_marker(period, marker_uptime, marker_offset)
                sequences.append(marker)
            else:
                volt_peak_to_peak = amplitude * VirtualAwg.__volt_to_millivolt
                sawtooth = Sequencer.make_sawtooth_wave(volt_peak_to_peak, period, width)
                sequences.append(sawtooth)
        self.sequence_gates(gate_names, sequences)

    def pulse_gates(self, gate_names, amplitudes, period, marker_uptime=0.2, marker_offset=0.0):
        if type(gate_names) == 'str':
            gate_names = [gate_names]
            amplitudes = [amplitudes]
        sequences = list()
        for gate_name, amplitude in zip(gate_names, amplitudes):
            (awg_nr, channel_nr, *marker_nr) = self.__gate_map[gate_name]
            if marker_nr:
                marker = Sequencer.make_marker(period, marker_uptime, marker_offset)
                sequences.append(marker)
            else:
                sawtooth = Sequencer.make_square_wave(amplitude, period)
                sequences.append(sawtooth)
        self.sequence_gates(gate_names, sequences)

    def sequence_gates(self, gate_names, sequences, do_upload=True):
        if type(gate_names) == 'str':
            gate_names = [gate_names]
            sequences = [sequences]
        if len(gate_names) != len(sequences):
            raise VirtualAwgError('Gate and sequence count do not match!')
        # check value gate_names and lengths of sequences per row...
        if do_upload:
            [awg.delete_waveforms() for awg in self.awgs]
        for awg_nr in self.__awg_range:
            channel_data = dict()
            waveform_data = dict()
            for (gate_name, sequence) in zip(gate_names, sequences):
                (nr, channel_nr, *marker_nr) = self.__gate_map[gate_name]
                if marker_nr:
                    continue
                if nr == awg_nr:
                    sequence_name = '{}_{}'.format(gate_name, sequence['name'])
                    sampling_rate = self.awgs[awg_nr].get_sampling_rate()
                    vpp_amplitude = self.awgs[awg_nr].get_setting('amplitudes')
                    awg_to_plunger = self.__hardware.parameters['awg_to_{}'.format(gate_name)].get()
                    self.sequence = sequence
                    scaling_ratio = 1e3 / awg_to_plunger / vpp_amplitude * 2.0
                    sample_data = Sequencer.get_data(sequence, sampling_rate)
                    sequence_data = sample_data * scaling_ratio
                    print('min {0}, max {1}, scaling {2}, values {3}'.format(min(sample_data),
                          max(sample_data), scaling_ratio, max(sample_data) * scaling_ratio))
                    self.pre_data = sample_data
                    self.seq_data = sequence_data
                    data_count = len(sequence_data)
                    waveform_data[sequence_name] = [sequence_data, np.zeros(data_count), np.zeros(data_count)]

                    # ...
                    for (gate_name_t, sequence_t) in zip(gate_names, sequences):
                        (awg_nr_t, channel_nr_t, *marker_nr_t) = self.__gate_map[gate_name_t]
                        if awg_nr_t == awg_nr and channel_nr_t == channel_nr and marker_nr_t:
                            marker_data = Sequencer.get_data(sequence_t, sampling_rate)
                            if len(marker_data) != data_count:
                                raise VirtualAwgError('Cannot add marker with unequal data-lenght!')
                            waveform_data[sequence_name][marker_nr_t[0]] = marker_data
                    channel_data.setdefault(channel_nr, []).append(sequence_name)
            if do_upload:
                self.awgs[awg_nr].upload_waveforms(list(waveform_data.keys()), list(waveform_data.values()))
            self.awgs[awg_nr].set_sequence(list(channel_data.keys()), list(channel_data.values()))

    def sweep_gates_2D(self, gate_names, amplitudes, period):
        pass

    def pulse_gates_2D(self, gate_names, amplitudes, period):
        pass

    def sequence_gates_2D(self, gate_names, sequences):
        pass


# UNITTESTS #

def test_init_HasNoErrors():
    from qtt.instrument_drivers.virtual_awg.awgs.simulated_awg import SimulatedAWG
    from unittest.mock import Mock
    awg_driver = Mock()
    awgs = [SimulatedAWG(awg_driver)]
    gate_map = {'P1': (0, 1), 'P2': (0, 2), 'digitizer_marker': (0, 1, 1)}
    _ = VirtualAwg(awgs, gate_map)
    # virtual_awg = VirtualAwg(awgs, gate_map)
    # self.assertEqual(awgs, virtual_awg.awgs)


"""
sweepparams = ['P1', 'P2']
sweepranges = [100, 100]
waveform, _ = self.station.awg.sweep_gate(sweepparams, sweepranges, period=1e-3)

"""
