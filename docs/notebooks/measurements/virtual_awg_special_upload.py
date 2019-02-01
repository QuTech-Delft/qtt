import numpy as np
import matplotlib.pyplot as plt

from qupulse.pulses import FunctionPT
from qupulse.pulses import SequencePT

from qcodes.utils.validators import Numbers
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import ManualParameter
from qcodes.instrument_drivers.tektronix.AWG5014 import Tektronix_AWG5014

from qtt.instrument_drivers.virtualAwg.sequencer import DataTypes
from qtt.instrument_drivers.virtualAwg.virtual_awg import VirtualAwg
from qtt.instrument_drivers.virtualAwg.sequencer import Sequencer

#%%

class HardwareSettings(Instrument):

    def __init__(self, name='settings'):
        """ Contains the quantum chip settings:
                awg_map: Relation between gate name and AWG number and AWG channel.
                awg_to_plunger: Scaling ratio between AWG output value and the voltoge on the gate.
        """
        super().__init__(name)
        awg_gates = {'X2': (0, 1), 'P7': (0, 2), 'P6': (0, 3), 'P5': (0, 4),
                     'P2': (1, 1), 'X1': (1, 2), 'P3': (1, 3), 'P4': (1, 4)}
        awg_markers = {'m4i_mk': (0, 4, 1)}
        self.awg_map = {**awg_gates, **awg_markers}

        for awg_gate in awg_gates:
            parameter_name = 'awg_to_{}'.format(awg_gate)
            parameter_label = '{} (factor)'.format(parameter_name)
            self.add_parameter(parameter_name, parameter_class=ManualParameter,
                               initial_value=1000.0, label=parameter_label, vals=Numbers(1, 1000))

#%% Initialize Tektronix AWG's

trigger_level_in_Volt = 0.5
clock_frequency_in_Hz = 1.0e7

address_awg1 = 'GPIB1::5::INSTR'
awg1 = Tektronix_AWG5014(name='awg1', address=address_awg1)
awg1.clock_freq(clock_frequency_in_Hz)
awg1.trigger_level(trigger_level_in_Volt)

#%% Initialize the Virtual AWG

def update_awg_settings(virtual_awg, sampling_rate, amplitude, marker_low, marker_high):
    for awg_number in range(len(virtual_awg.awgs)):
        virtual_awg.update_setting(awg_number, 'sampling_rate', sampling_rate)
        virtual_awg.update_setting(awg_number, 'amplitudes', amplitude)
        virtual_awg.update_setting(awg_number, 'marker_low', marker_low)
        virtual_awg.update_setting(awg_number, 'marker_high', marker_high)

awg_number = 0
settings = HardwareSettings()
virtual_awg = VirtualAwg([awg1], settings)

uptime_in_seconds = 1.0e-5
marker_delay_in_sec = 3.5e-5

virtual_awg.update_digitizer_marker_settings(uptime_in_seconds, marker_delay_in_sec)

output_amplitude = 0.5
marker_low_level = 0.0
marker_high_level = 2.6

update_awg_settings(virtual_awg, clock_frequency_in_Hz, output_amplitude, marker_low_level, marker_high_level)

#%% Create some waveform

sec_to_ns = 1.0e9

mV_amplitude = 25
sec_period = 1.0e-3
sine_decay = 5e5
sine_period = 2 * np.pi * 1e-2 * sec_period

pulse_function = FunctionPT('alpha*exp(-t/tau)*sin(phi*t)', 'duration')
input_variables = {'alpha': mV_amplitude, 'tau': sine_decay, 'phi': sine_period, 'duration': sec_period * sec_to_ns}

other = (pulse_function, input_variables)
sine_waveform = {'name': 'test', 'wave': SequencePT(*(other,)), 'type': DataTypes.QU_PULSE}


#%% Create some marker

marker = virtual_awg._VirtualAwg__make_markers(sec_period)


#%%

sampling_rate = virtual_awg.awgs[awg_number].retrieve_sampling_rate()
vpp_amplitude = virtual_awg.awgs[awg_number].retrieve_gain()

waveform = Sequencer.get_data(sine_waveform, sampling_rate)
marker_data = Sequencer.get_data(marker['m4i_mk'], sampling_rate)
empty_data = np.zeros(len(marker_data))

#%%

scaling_ratio_X2 = 2 / (settings.awg_to_X2() * vpp_amplitude)
waveform1 = waveform * scaling_ratio_X2

scaling_ratio_P7 = 2 / (settings.awg_to_P7() * vpp_amplitude)
waveform2 = waveform * scaling_ratio_X2

#%%

waveform_data = {
        'MyWave1': [waveform1 * scaling_ratio_X2, marker_data, empty_data],
        'MyWave2': [waveform2 * scaling_ratio_P7, marker_data, empty_data]}
virtual_awg._upload_waveforms(awg_number, waveform_data)


#%%

channel_numbers = [1, 2]
waveform_names = [['MyWave1', 'MyWave2'], ['MyWave1', 'MyWave2']]
virtual_awg._set_sequence_order(awg_number, channel_numbers, waveform_names)

#%%