import numpy as np

from qupulse.pulses import FunctionPulseTemplate
from qupulse.pulses import SequencePulseTemplate

from qcodes.utils.validators import Numbers
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.instrument_drivers.Spectrum.M4i import M4i
from qcodes.instrument_drivers.Tektronix.AWG5014 import Tektronix_AWG5014

from qtt.instrument_drivers.virtual_awg_new import VirtualAwg
from qtt.instrument_drivers.virtualAwg.sequencer import DataTypes
from qtt.measurements.scans import measuresegment as measure_segment

import pyspcm


class QuantumDeviceSettings(Instrument):

    def __init__(self, name='settings'):
        """ Contains the quantum chip settings:
                awg_map: Relation betweem gate and AWG number and AWG channel.
                awg_to_plunger: Scaling ratio between AWG output value and the voltoge on the gate.
        """
        super().__init__(name)
        awg_gates = {'X2': (0, 1), 'P7': (0, 2), 'P6': (0, 3), 'P5': (0, 4),
                     'P2': (1, 1), 'X1': (1, 2), 'P3': (1, 3), 'P4': (1, 4)}
        awg_markers = {'m4i_mk': (0, 4, 1), 'awg_mk': (0, 4, 2)}
        self.awg_map = {**awg_gates, **awg_markers}

        for awg_gate in awg_gates:
            parameter_name = 'awg_to_{}'.format(awg_gate)
            parameter_label = '{} (factor)'.format(parameter_name)
            self.add_parameter(awg_gate, parameter_class=ManualParameter, initial_value=1.0,
                               label=parameter_label, vals=Numbers(1, 400))


# Initialize digitizer
digitizer = M4i(name='digitizer')

sample_rate_in_Hz = 61035
digitizer.sample_rate(sample_rate_in_Hz)

timeout_in_ms = 2.0e4
digitizer.timeout(timeout_in_ms)

millivolt_range = 1000
digitizer.initialize_channels(mV_range=millivolt_range)

external_clock_mode = pyspcm.SPC_CM_EXTREFCLOCK
digitizer.clock_mode(external_clock_mode)

reference_clock_10mHz = int(1e7)
digitizer.reference_clock(reference_clock_10mHz)


# Initialize Tektronix AWG's
clock_frequency_in_Hz = 1.0e7
trigger_level_in_Volt = 0.5
output_amplitude = 1.0

address_awg1 = 'TCPIP0::169.254.182.194::inst0::INSTR'
awg1 = Tektronix_AWG5014(name='awg1', address=address_awg1)
awg1.clock_freq(clock_frequency_in_Hz)
awg1.trigger_level(trigger_level_in_Volt)

address_awg2 = 'TCPIP0::169.254.182.194::inst1::INSTR'
awg2 = Tektronix_AWG5014(name='awg2', address=address_awg2)
awg2.clock_freq(clock_frequency_in_Hz)
awg2.trigger_level(trigger_level_in_Volt)

awgs = [awg1, awg2]


# Initialize the device settings
settings = QuantumDeviceSettings()


# Initialize the virtual AWG
virtual_awg = VirtualAwg(awgs, settings)

virtual_awg.digitizer_marker_delay.set(1.0e-3)
virtual_awg.digitizer_marker_uptime(5.0e-8)

virtual_awg.awg_marker_delay.set(0.0)
virtual_awg.awg_marker_uptime(5.0e-8)

virtual_awg.update_setting(0, 'sampling_rate', clock_frequency_in_Hz)
virtual_awg.update_setting(0, 'amplitudes', output_amplitude)

virtual_awg.update_setting(1, 'sampling_rate', clock_frequency_in_Hz)
virtual_awg.update_setting(1, 'amplitudes', output_amplitude)


# Example sweep_gates with readout.
sec_period = 1.0e-6
mV_sweep_range = 100
gates = {'P5': 1, 'P6': 2}
sweep_data = virtual_awg.sweep_gates(gates, mV_sweep_range, sec_period)

virtual_awg.enable_outputs(['P5', 'P6'])
virtual_awg.run()

number_of_averages = 100
instrument_handle_numbers = [1]
data = measure_segment(sweep_data, number_of_averages, digitizer, instrument_handle_numbers)

virtual_awg.disable_outputs(['P5', 'P6'])
virtual_awg.stop()


# Example pulse_gates with readout.
sec_period = 1.0e-5
mV_sweep_range = 250
gates = {'P5': 1, 'P6': 2}
sweep_data = virtual_awg.pulse_gates(gates, mV_sweep_range, sec_period)

virtual_awg.enable_outputs(['P5', 'P6'])
virtual_awg.run()

number_of_averages = 100
instrument_handle_numbers = [1]
data = measure_segment(sweep_data, number_of_averages, digitizer, instrument_handle_numbers)

virtual_awg.disable_outputs(['P5', 'P6'])
virtual_awg.stop()


# Example sweep_gates_2D with readout.
sec_period = 1.0e-6
mV_sweep_ranges = [100, 100]
resolution = [100, 100]
gates = [{'P5': 1}, {'P6': 2}]
sweep_data = virtual_awg.sweep_gates_2D(gates, mV_sweep_ranges, sec_period, resolution)

virtual_awg.enable_outputs(['P5', 'P6'])
virtual_awg.run()

number_of_averages = 100
instrument_handle_numbers = [1]
data = measure_segment(sweep_data, number_of_averages, digitizer, instrument_handle_numbers)

virtual_awg.disable_outputs(['P5', 'P6'])
virtual_awg.stop()


# Example sequence_gates with readout.
sec_period = 1.0e-5
pulse_function = FunctionPulseTemplate('exp(-t/tau)*sin(phi*t)', 'duration')
input_variables = {'tau': 4, 'phi': 8, 'duration': 8 * np.pi}
sequence_data = SequencePulseTemplate(pulse_function, input_variables)
sequence = {'name': 'decay_wave', 'wave': sequence_data, 'type': DataTypes.QU_PULSE}

gates = {'P5': 1, 'P6': 2}
sweep_data = virtual_awg.sequence_gates(gates, sequence, sec_period)

virtual_awg.enable_outputs(['P5', 'P6'])
virtual_awg.run()

number_of_averages = 100
instrument_handle_numbers = [1]
data = measure_segment(sweep_data, number_of_averages, digitizer, instrument_handle_numbers)

virtual_awg.disable_outputs(['P5', 'P6'])
virtual_awg.stop()

number_of_averages = 100
instrument_handle_numbers = [1]
data = measure_segment(sweep_data, number_of_averages, digitizer, instrument_handle_numbers)

virtual_awg.disable_outputs(['P5', 'P6'])
virtual_awg.stop()
