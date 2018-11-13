import numpy as np
import matplotlib.pyplot as plt

from qupulse.pulses import FunctionPT
from qupulse.pulses import SequencePT

from qcodes.utils.validators import Numbers
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.instrument_drivers.Spectrum.M4i import M4i
from qcodes.instrument_drivers.tektronix.AWG5014 import Tektronix_AWG5014

from qtt.instrument_drivers.virtualAwg.sequencer import DataTypes
from qtt.instrument_drivers.virtualAwg.virtual_awg import VirtualAwg
from qtt.measurements.scans import measuresegment as measure_segment


def plot_data(digitizer_data):
    plt.figure(); 
    plt.clf(); 
    plt.plot(digitizer_data.flatten(),'.b')
    plt.show()


class QuantumDeviceSettings(Instrument):

    def __init__(self, name='settings'):
        """ Contains the quantum chip settings:
                awg_map: Relation betweem gate and AWG number and AWG channel.
                awg_to_plunger: Scaling ratio between AWG output value and the voltoge on the gate.
        """
        super().__init__(name)
        awg_gates = {'X2': (0, 1), 'P7': (0, 2), 'P6': (0, 3), 'P5': (0, 4),
                     'P2': (1, 1), 'X1': (1, 2), 'P3': (1, 3), 'P4': (1, 4)}
        awg_markers = {'m4i_mk': (0, 4, 1)} #, 'awg_mk': (0, 4, 2)}
        self.awg_map = {**awg_gates, **awg_markers}

        for awg_gate in awg_gates:
            parameter_name = 'awg_to_{}'.format(awg_gate)
            parameter_label = '{} (factor)'.format(parameter_name)
            self.add_parameter(parameter_name, parameter_class=ManualParameter,
                               initial_value=1.0, label=parameter_label, vals=Numbers(1, 400))


#%%


# Initialize digitizer
digitizer = M4i(name='digitizer')

sample_rate_in_Hz = 2e6
digitizer.sample_rate(sample_rate_in_Hz)

timeout_in_ms = 10 * 1000
digitizer.timeout(timeout_in_ms)

millivolt_range = 2000
digitizer.initialize_channels(mV_range=millivolt_range)

import pyspcm
external_clock_mode = pyspcm.SPC_CM_EXTREFCLOCK
digitizer.clock_mode(external_clock_mode)

reference_clock_10mHz = int(1e7)
digitizer.reference_clock(reference_clock_10mHz)


# Initialize Tektronix AWG's
trigger_level_in_Volt = 0.5
clock_frequency_in_Hz = 1.0e7

address_awg1 = 'GPIB0::5::INSTR'
awg1 = Tektronix_AWG5014(name='awg1', address=address_awg1)
awg1.clock_freq(clock_frequency_in_Hz)
awg1.trigger_level(trigger_level_in_Volt)

address_awg2 = 'GPIB0::26::INSTR'
awg2 = Tektronix_AWG5014(name='awg2', address=address_awg2)
awg2.clock_freq(clock_frequency_in_Hz)
awg2.trigger_level(trigger_level_in_Volt)


#%% Initialize the device settings and the virtual AWG
settings = QuantumDeviceSettings()
virtual_awg = VirtualAwg([awg1, awg2], settings)

virtual_awg.digitizer_marker_delay.set(3.5e-5)
virtual_awg.digitizer_marker_uptime(1.0e-4)

#virtual_awg.awg_marker_delay.set(0.0)
#virtual_awg.awg_marker_uptime(5.0e-7)

output_amplitude = 0.5
marker_low_level = 0.0
marker_high_level = 2.6

for awg_number in range(len(virtual_awg.awgs)):
    #virtual_awg.update_setting(awg_number, 'sampling_rate', clock_frequency_in_Hz)
    virtual_awg.update_setting(awg_number, 'amplitudes', output_amplitude)
    virtual_awg.update_setting(awg_number, 'marker_low', marker_low_level)
    virtual_awg.update_setting(awg_number, 'marker_high', marker_high_level)

#%% Example sweep_gates with readout.

sec_period = 1.0e-3
mV_sweep_range = 50
gates = {'X2': 2}
sweep_data = virtual_awg.sweep_gates(gates, mV_sweep_range, sec_period)

virtual_awg.enable_outputs(['X2'])
virtual_awg.run()

number_of_averages = 100
readout_channels = [0]
data = measure_segment(sweep_data, number_of_averages, digitizer, readout_channels)

virtual_awg.disable_outputs(['X2'])
virtual_awg.stop()

plot_data(data)

#%% Example pulse_gates with readout.

sec_period = 1.0e-3
mV_sweep_range = 100
gates = {'X2': 1}
sweep_data = virtual_awg.pulse_gates(gates, mV_sweep_range, sec_period)
sweep_data['width'] = 1.0

virtual_awg.enable_outputs(['X2'])
virtual_awg.run()

number_of_averages = 100
readout_channels = [0]
data = measure_segment(sweep_data, number_of_averages, digitizer, readout_channels, process=False)

virtual_awg.disable_outputs(['X2'])
virtual_awg.stop()

plot_data(data)

#%% Example sweep_gates_2D with readout.
sec_period = 1.0e-3
mV_sweep_ranges = [100, 100]
resolution = [100, 100]
gates = [{'X2': 1}, {'P7': 1}]
sweep_data = virtual_awg.sweep_gates_2d(gates, mV_sweep_ranges, sec_period, resolution)

virtual_awg.enable_outputs(['X2', 'P7'])
virtual_awg.run()

number_of_averages = 100
readout_channels = [0]
data = measure_segment(sweep_data, number_of_averages, digitizer, readout_channels)

virtual_awg.disable_outputs(['X2', 'P7'])
virtual_awg.stop()

plot_data(data)

#%% Example sequence_gates with readout.

sec_to_ns = 1.0e9

mV_amplitude = 25
sec_period = 1.0e-3

sine_period = 2 * np.pi * 1e-2 * sec_period
sine_decay = 5e5

pulse_function = FunctionPT('alpha*exp(-t/tau)*sin(phi*t)', 'duration')
input_variables = {'alpha': mV_amplitude, 'tau': sine_decay, 'phi': sine_period, 'duration': sec_period * sec_to_ns}
sequence_data = (pulse_function, input_variables)
sequence = {'name': 'test', 'wave': SequencePT(*(sequence_data,)), 'type': DataTypes.QU_PULSE}

sequences = {'X2': sequence}
sequences.update(virtual_awg._VirtualAwg__make_markers(sec_period))

sweep_data = virtual_awg.sequence_gates(sequences)
sweep_data.update({
    'width': 1.0,
    'period': sec_period,
    'samplerate': virtual_awg.awgs[0].retrieve_setting('sampling_rate'),
    'markerdelay': virtual_awg.digitizer_marker_delay()})

virtual_awg.enable_outputs(['X2'])
virtual_awg.run()

number_of_averages = 100
readout_channels = [0]
data = measure_segment(sweep_data, number_of_averages, digitizer, readout_channels)

virtual_awg.disable_outputs(['X2'])
virtual_awg.stop()

plot_data(data)


#%% Close all devices

virtual_awg.close()
settings.close()

awg1.close()
awg2.close()
digitizer.close()
