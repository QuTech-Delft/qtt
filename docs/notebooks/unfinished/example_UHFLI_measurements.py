#%% Importing the modules and defining functions
import os

from matplotlib import pyplot as plt
from qcodes.utils.validators import Numbers
from qcodes import Station, Instrument, ManualParameter
from qcodes.instrument_drivers.ZI.ZIHDAWG8 import WARNING_ANY
from qilib.configuration_helper import InstrumentAdapterFactory

from qtt.instrument_drivers.virtual_instruments import VirtualIVVI
from qtt.instrument_drivers.virtualAwg.virtual_awg import VirtualAwg
from qtt.measurements.acquisition import UHFLIScopeReader, UHFLIStimulus, load_configuration
from qtt.measurements.scans import scan1D, scanjob_t
from qtt.measurements.videomode import VideoMode


def stimulus_enabled(enabled_state, demod_number=1, demod_channel=1, output_amplitude=0.1, output_channel=1):
    stimulus.set_signal_output_amplitude(demod_channel, demod_number, output_amplitude)
    stimulus.set_demodulation_enabled(demod_channel, enabled_state)
    stimulus.set_signal_output_enabled(demod_channel, demod_number, enabled_state)
    stimulus.set_output_enabled(output_channel, enabled_state)

def stimulus_frequency(frequency, demod_channel=1):
    stimulus.set_oscillator_frequency(demod_channel, frequency)

def scope_settings(period=None, sample_rate=27e3, input_channel=1, input_channel_attribute='Demod 1 R',
                   input_range=1, limits=(0.0, 1.0), trigger_enabled=False, trigger_level=0.5):
    nearest_sample_rate = scope_reader.get_nearest_sample_rate(sample_rate)
    scope_reader.sample_rate = nearest_sample_rate

    scope_reader.enabled_channels = (input_channel, )
    scope_reader.input_range = tuple([input_range] * 2)
    scope_reader.set_channel_limits(input_channel, *limits)
    scope_reader.set_input_signal(input_channel, input_channel_attribute)
    if period:
        scope_reader.period = period

    scope_reader.trigger_enabled = trigger_enabled
    scope_reader.trigger_channel = 'Trig Input 1'
    scope_reader.trigger_level = trigger_level
    scope_reader.trigger_slope = 'Rise'
    scope_reader.trigger_delay = 0



## Settings

class HardwareSettings(Instrument):

    def __init__(self, name='settings'):
        super().__init__(name)
        awg_gates = {'P1': (0, 0), 'P2': (0, 1)}
        awg_markers = {'m4i_mk': (0, 4, 0)}
        self.awg_map = {**awg_gates, **awg_markers}

        for awg_gate in self.awg_map:
            parameter_name = 'awg_to_{}'.format(awg_gate)
            parameter_label = '{} (factor)'.format(parameter_name)
            self.add_parameter(parameter_name, parameter_class=ManualParameter,
                               initial_value=1000, label=parameter_label, vals=Numbers(1, 1000))


working_directory = "<clone_dir>\\qtt\\docs\\notebooks\\unfinished"

## Lock-in Amplifier

UHFLI_id = 'dev2338'
file_path_UHFLI = os.path.join(working_directory, f'UHFLI_{UHFLI_id}.dat')
UHFLI_configuration = load_configuration(file_path_UHFLI)

stimulus = UHFLIStimulus(UHFLI_id)
stimulus.initialize(UHFLI_configuration)
scope_reader = UHFLIScopeReader(UHFLI_id)


## AWG

HDAWG8_id = 'dev8048'
file_path_HDAWG8 = os.path.join(working_directory, f'HDAWG8_{HDAWG8_id}.dat')
HDAWG8_configuration = load_configuration(file_path_HDAWG8)

awg_adapter = InstrumentAdapterFactory.get_instrument_adapter('ZIHDAWG8InstrumentAdapter', HDAWG8_id)
awg_adapter.apply(HDAWG8_configuration)

awg_adapter.instrument.warnings_as_errors.append(WARNING_ANY)

external_clock = 1
awg_adapter.instrument.system_clocks_referenceclock_source(external_clock)

grouping_1x8 = 2
awg_adapter.instrument.set_channel_grouping(grouping_1x8)

output4_marker1 = 4
awg_adapter.instrument.triggers_out_4_source(output4_marker1)

awg_sampling_rate_586KHz = 12
awg_adapter.instrument.awgs_0_time(awg_sampling_rate_586KHz)


## Virtual AWG

settings = HardwareSettings()
virtual_awg = VirtualAwg([awg_adapter.instrument], settings)

marker_delay = 16e-6
virtual_awg.digitizer_marker_delay(marker_delay)

marker_uptime = 1e-2
virtual_awg.digitizer_marker_uptime(marker_uptime)


## Station

gates = VirtualIVVI('gates', gates=['P1', 'P2'], model=None)
station = Station(virtual_awg, scope_reader.adapter.instrument, awg_adapter.instrument, gates)

#%% Enable the stimulus

demod_channel = 1
stimulus_enabled(True, demod_channel=demod_channel)


#%% Sensing a resonance

demod_parameter = 'R'

scanjob = scanjob_t()
scanjob.add_sweep(param=stimulus.set_oscillator_frequency(demod_channel), start=140e6, end=180e6, step=0.2e6)
scanjob.add_minstrument(scope_reader.acquire_single_sample(demod_channel, demod_parameter, partial=True))

data_set = scan1D(station, scanjob)

plt.clf()
plt.plot(data_set.arrays[f'oscillator{demod_channel}_freq'],
         data_set.arrays[f'demod{demod_channel}_{demod_parameter}'])
plt.show()


#%% 1D readout

period = 1.0
frequency = 154.8e6

scope_settings(period)
stimulus_frequency(frequency)

scope_reader.start_acquisition()
data_arrays = scope_reader.acquire(number_of_averages=1)
scope_reader.stop_acquisition()

plt.clf()
plt.plot(data_arrays[0].set_arrays[0], data_arrays[0])
plt.show()


#%% 2D video mode

sample_rate = 220e3

scope_settings(sample_rate=sample_rate, trigger_enabled=True)

resolution = [96, 96]
sweep_gates = [{'P1': 1}, {'P2': 1}]
sweep_ranges = [1000, 1000]
scope = (scope_reader, [1], ['Demod 1 R'])

vm = VideoMode(station, sweep_gates, sweep_ranges, minstrument=scope, resolution=resolution)
vm.updatebg()


#%% Disable the stimulus

stimulus_enabled(False, demod_channel=demod_channel)
