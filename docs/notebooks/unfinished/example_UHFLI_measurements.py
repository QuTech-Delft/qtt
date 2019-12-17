#%% Importing the modules and defining functions
import os
from typing import Optional, Tuple, Dict

from matplotlib import pyplot as plt
from qcodes import Instrument, ManualParameter, Station
from qcodes.instrument_drivers.ZI.ZIHDAWG8 import WARNING_ANY
from qcodes.utils.validators import Numbers
from qilib.configuration_helper import InstrumentAdapterFactory

from qtt.instrument_drivers.virtual_instruments import VirtualIVVI
from qtt.instrument_drivers.virtualAwg.virtual_awg import VirtualAwg
from qtt.measurements.acquisition import UHFLIScopeReader, UHFLIStimulus, load_configuration
from qtt.measurements.scans import scan1D, scanjob_t
from qtt.measurements.videomode import VideoMode


def update_stimulus(is_enabled: bool, signal_output: int = 1, signal_input: Optional[int] = None,
                    amplitude: float = 0.1, oscillator: int = 1) -> None:
    """ Sets the enabled status of a demodulator and connects an oscillator to a demodulator.
        After that sets for a signal output, the amplitude and enabled status and finally to  of the UHFLI.

        Note that each oscillator and demodulator is connected one-to-one in this function.
        Also the output signal and demodulator input signal is connected one-to-one.

    Args:
        is_enabled: True to enable and False to disable.
        signal_output: One of the two outputs on the device.
        signal_input: One of the two inputs on the device.
        amplitude: Amplitude in volts, allowed values are 0.0 - 1.5 V.
        demodulator: Which demodulator used to connect to the signal output to.
    """
    demodulator = oscillator
    if not signal_input:
        signal_input = signal_output

    stimulus.connect_oscillator_to_demodulator(oscillator, demodulator)
    stimulus.set_demodulator_signal_input(demodulator, signal_input)
    stimulus.set_demodulation_enabled(demodulator, is_enabled)

    stimulus.set_signal_output_amplitude(signal_output, demodulator, amplitude)
    stimulus.set_signal_output_enabled(signal_output, demodulator, is_enabled)
    stimulus.set_output_enabled(signal_output, is_enabled)


def set_stimulus_oscillator_frequency(frequency: float, oscillator: int = 1) -> None:
    """ Sets the frequency of an oscillator in the UHFLI.

        Note this function only works with the UHFLI multifrequency option enabled (MF option).

    Args:
        frequency: The set frequency of the oscillator in Hz.
        oscillator: The oscillator (1 - 8) to set the frequency on.
    """
    stimulus.set_oscillator_frequency(oscillator, frequency)


def update_scope(period: Optional[float] = None, sample_rate: Optional[float] = 27e3,
                 input_channels : Optional[Dict[str, int]] = None,
                 input_ranges: Optional[Tuple[float, float]] = (1.0, 1.0),
                 limits: Optional[Tuple[float, float]] = (0.0, 1.0),
                 trigger_enabled: Optional[bool] = False, trigger_level: Optional[float] = 0.5):
    """ Updates the settings of the UHFLI scope module needed for readout.

    Args:
        period: The measuring period of the acquisition in seconds.
        sample_rate: The sample rate of the acquisition device in samples per second.
        input_channels: A dictionary containing the signal names and input channels, e.g. {'Demod 1 R': 1}.
        input_ranges: The gain in Volt of the analog input amplifier for both input channels.
        limits: The limits in Volt of the scope full scale range.
        trigger_enabled: Will enable the external triggering on 'Trigger input 1' if True.
        trigger_level: The level of the trigger in Volt.
    """
    if period is not None:
        scope_reader.period = period

    if input_channels is None:
        input_channels = {'Demod 1 R': 1}

    nearest_sample_rate = scope_reader.get_nearest_sample_rate(sample_rate)
    scope_reader.sample_rate = nearest_sample_rate

    scope_reader.input_range = input_ranges
    unique_channels = set(input_channels.values())
    scope_reader.enabled_channels = unique_channels

    [scope_reader.set_channel_limits(channel, *limits) for channel in unique_channels]
    [scope_reader.set_input_signal(channel, attribute) for (attribute, channel) in input_channels.items()]

    scope_reader.trigger_enabled = trigger_enabled
    scope_reader.trigger_channel = 'Trig Input 1'
    scope_reader.trigger_level = trigger_level
    scope_reader.trigger_slope = 'Rise'
    scope_reader.trigger_delay = 0


#  Settings

class HardwareSettings(Instrument):

    def __init__(self, name='settings'):
        super().__init__(name)
        awg_gates = {'P1': (0, 0), 'P2': (0, 1)}
        awg_markers = {'m4i_mk': (0, 0, 0)}
        self.awg_map = {**awg_gates, **awg_markers}

        for awg_gate in self.awg_map:
            parameter_name = 'awg_to_{}'.format(awg_gate)
            parameter_label = '{} (factor)'.format(parameter_name)
            self.add_parameter(parameter_name, parameter_class=ManualParameter,
                               initial_value=1000, label=parameter_label, vals=Numbers(1, 1000))


working_directory = "<work_directory>\\qtt\\docs\\notebooks\\unfinished"

#  Lock-in Amplifier

uhfli_id = 'dev2338'
file_path_UHFLI = os.path.join(working_directory, f'UHFLI_{uhfli_id}.dat')
uhfli_configuration = load_configuration(file_path_UHFLI)

stimulus = UHFLIStimulus(uhfli_id)
stimulus.initialize(uhfli_configuration)
scope_reader = UHFLIScopeReader(uhfli_id)

scope_reader.adapter.instrument.external_clock_enabled('ON')

#  AWG

hdawg8_id = 'dev8048'
file_path_HDAWG8 = os.path.join(working_directory, f'HDAWG8_{hdawg8_id}.dat')
hdawg8_configuration = load_configuration(file_path_HDAWG8)

awg_adapter = InstrumentAdapterFactory.get_instrument_adapter('ZIHDAWG8InstrumentAdapter', hdawg8_id)
awg_adapter.apply(hdawg8_configuration)

awg_adapter.instrument.warnings_as_errors.append(WARNING_ANY)

external_clock = 1
awg_adapter.instrument.system_clocks_referenceclock_source(external_clock)

grouping_1x8 = 2
awg_adapter.instrument.set_channel_grouping(grouping_1x8)

output1_marker1 = 4
awg_adapter.instrument.triggers_out_0_source(output1_marker1)

awg_sampling_rate_586KHz = 12
awg_adapter.instrument.awgs_0_time(awg_sampling_rate_586KHz)


#  Virtual AWG

settings = HardwareSettings()
virtual_awg = VirtualAwg([awg_adapter.instrument], settings)

marker_delay = 16e-6
virtual_awg.digitizer_marker_delay(marker_delay)

marker_uptime = 1e-2
virtual_awg.digitizer_marker_uptime(marker_uptime)


#  Station

gates = VirtualIVVI('gates', gates=['P1', 'P2'], model=None)
station = Station(virtual_awg, scope_reader.adapter.instrument, awg_adapter.instrument, gates)

#%% Enable the stimulus

demodulator = 1
signal_output = 1
oscillator = demodulator

update_stimulus(is_enabled=True, signal_output=signal_output, amplitude=0.1, oscillator=oscillator)

#%% Sensing a resonance

signal_input = signal_output

scanjob = scanjob_t(wait_time_startscan=1)
scanjob.add_sweep(param=stimulus.set_oscillator_frequency(oscillator), start=140e6, end=180e6, step=0.2e6)
scanjob.add_minstrument([scope_reader.acquire_single_sample(demodulator, 'R', partial=True),
                         scope_reader.acquire_single_sample(demodulator, 'x', partial=True),
                         scope_reader.acquire_single_sample(demodulator, 'y', partial=True)])

data_set = scan1D(station, scanjob)

plt.clf()
plt.plot(data_set.arrays[f'oscillator{demodulator}_freq'], data_set.arrays[f'demod{demodulator}_R'])
plt.plot(data_set.arrays[f'oscillator{demodulator}_freq'], data_set.arrays[f'demod{demodulator}_x'])
plt.plot(data_set.arrays[f'oscillator{demodulator}_freq'], data_set.arrays[f'demod{demodulator}_y'])
plt.show()


#%% 1D readout

period = 1.0
frequency = 154.8e6

update_scope(period)
set_stimulus_oscillator_frequency(frequency)

scope_reader.start_acquisition()
data_arrays = scope_reader.acquire(number_of_averages=1)
scope_reader.stop_acquisition()

plt.clf()
plt.plot(data_arrays[0].set_arrays[0], data_arrays[0])
plt.show()


#%% 2D video mode

sample_rate = 220e3

update_scope(sample_rate = sample_rate, trigger_enabled = True)

resolution = [96, 96]
sweep_gates = [{'P1': 1}, {'P2': 1}]
sweep_ranges = [1000, 1000]
scope = (scope_reader, [1], ['Demod 1 R'])

vm = VideoMode(station, sweep_gates, sweep_ranges, minstrument=scope, resolution=resolution)
vm.updatebg()


#%% Disable the stimulus

update_stimulus(is_enabled=False, signal_output=signal_output, oscillator=oscillator)
