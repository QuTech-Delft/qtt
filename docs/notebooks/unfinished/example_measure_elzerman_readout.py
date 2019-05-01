from qtt.measurements.acquisition import UHFLIScopeReader, UHFLIStimulus
from matplotlib import pyplot as plt


# INITIALIZE THE SCOPE READER

device_id = 'dev2338'
time_period = 2
trigger_enabled = True
input_channel = 1
input_channel_attribute = 'Demod 1 R'
scope_reader = UHFLIScopeReader(address=device_id)
sampling_rate = scope_reader.get_nearest_sample_rate(27e3)
scope_reader.period = time_period
scope_reader.trigger_enabled = trigger_enabled
scope_reader.enabled_channels = (input_channel, )
scope_reader.set_input_signal(input_channel, input_channel_attribute)
scope_reader.trigger_enabled = trigger_enabled
scope_reader.trigger_channel = 'Trig Input 1'
scope_reader.trigger_level = 0.100
scope_reader.trigger_slope = 'Rise'
scope_reader.trigger_delay = 0


# INITIALIZE THE STIMULUS

output_channel = 1
demodulator = 1
demodulator_channel = 1
output_amplitude = 0.4
oscillator_frequency = 154.8e6
stimulus = UHFLIStimulus(address=device_id)
stimulus.set_signal_output_amplitude(demodulator_channel, demodulator, output_amplitude)
stimulus.set_demodulation_enabled(demodulator_channel, True)
stimulus.set_signal_output_enabled(demodulator_channel, demodulator, True)
stimulus.set_output_enabled(output_channel, True)
stimulus.set_oscillator_frequency(demodulator_channel, oscillator_frequency)


# READOUT THE SCOPE AND PLOT THE DATA

scope_reader.start_acquisition()
records = scope_reader.acquire(number_of_records=1)
scope_reader.stop_acquisition()
plt.plot(records[0])
plt.show()
print(records)


# DISABLE CHANNELS AND OUTPUTS

stimulus.set_demodulation_enabled(demodulator_channel, False)
stimulus.set_signal_output_enabled(demodulator_channel, demodulator, False)
stimulus.set_output_enabled(output_channel, False)
