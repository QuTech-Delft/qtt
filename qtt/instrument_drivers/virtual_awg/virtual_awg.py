import logging
import numpy as np

from qtt.instrument_drivers.virtual_awg.sequencer import Sequencer


class VirtualAwgError(Exception):
    """Exception for a specific error related to the virtual AWG."""


class VirtualAwg:

    def __init__(self, awgs, gate_map, logger=logging):
        self.__awg_range = range(0, len(awgs))
        self.__awg_count = len(awgs)
        self.awgs = awgs
        self.__gate_map = gate_map
        self.__logger = logger

    def run_awgs(self):
        [awg.run() for awg in self.awgs]

    def stop_awgs(self):
        [awg.stop() for awg in self.awgs]

    def reset_awgs(self):
        [awg.reset() for awg in self.awgs]

    def enable_outputs(self, awg_nr, channels):
        if awg_nr not in self.__awg_range:
            raise VirtualAwgError('Invalid AWG nr {}!'.format(awg_nr))
        self.awgs[awg_nr].enable_outputs(channels)

    def disable_outputs(self, awg_nr, channels):
        if awg_nr not in self.__awg_range:
            raise VirtualAwgError('Invalid AWG nr {}!'.format(awg_nr))
        self.awgs[awg_nr].disable_outputs(channels)

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

    def sweep_gates(self, gate_names, amplitudes, period):
        sequences = list()
        for amplitude in amplitudes:
            sawtooth = Sequencer.make_sawtooth_wave(amplitude, period)
            sequences.append(sawtooth)
        self.sequence_gates(gate_names, sequences)

    def pulse_gates(self, gate_names, amplitudes, period):
        sequences = list()
        for amplitude in amplitudes:
            sawtooth = Sequencer.make_square_wave(amplitude, period)
            sequences.append(sawtooth)
        self.sequence_gates(gate_names, sequences)

    def sequence_gates(self, gate_names, sequences, do_upload=True):
        if len(gate_names) != len(sequences):
            raise VirtualAwgError('Gate and sequence count do not match!')
        if len(gate_names) == 0:
            raise VirtualAwgError('Cannot upload an empty set of gates/sequences.')
        # check value gate_names and lengths of sequences per row...
        if do_upload:
            [awg.delete_waveforms() for awg in self.awgs]
        for awg_nr in self.__awg_range:
            channel_data = dict()
            sequence_data = dict()
            for (gate_name, sequence) in zip(gate_names, sequences):
                (nr, channel_nr) = gate_names[gate_name]
                if nr == awg_nr:
                    sequence_name = '{}_{}'.format(gate_name, sequence['NAME'])
                    sequence_data[sequence_name] = sequence
                    channel_data.setdefault(channel_nr, []).append(sequence_name)
            if do_upload:
                self.awgs[awg_nr].upload_waveforms(sequence_data.keys(), sequence_data.values())
            self.awgs[awg_nr].set_sequence(channel_data.keys(), list(channel_data.values()))

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