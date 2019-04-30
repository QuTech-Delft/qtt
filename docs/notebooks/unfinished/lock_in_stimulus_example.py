from qtt.measurements.acquisition import UHFLIStimulus

lock_in_stimulus = UHFLIStimulus('dev2338')


channel = 2
oscillator_frequency = 145e6
amplitude = 0.1
demodulator = 1
demodulator_enabled = True
signal_output_enabled = True
output_enabled = True
output = 1

lock_in_stimulus.set_oscillator_frequency(channel, oscillator_frequency)
lock_in_stimulus.set_signal_output_amplitude(channel, demodulator, amplitude)
lock_in_stimulus.set_demodulation_enabled(channel, demodulator_enabled)
lock_in_stimulus.set_signal_output_enabled(channel, demodulator, signal_output_enabled)
lock_in_stimulus.set_output_enabled(output, output_enabled)
