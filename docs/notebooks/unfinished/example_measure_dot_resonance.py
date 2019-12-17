import matplotlib.pyplot as plt
from qcodes import Station
from qilib.utils import PythonJsonStructure

import qtt
from qtt.measurements.acquisition import UHFLIScopeReader, UHFLIStimulus


class FakeGates:
    def allvalues(self):
        return {}


# Setup UHFLIScopeReader
device_id = 'dev2338'
configuration = PythonJsonStructure(demod1_timeconstant={'value': 0.0008104579756036401},
                                    signal_input1_impedance={'value': 50},
                                    signal_output1_imp50={'value': 'ON'})
scope_reader = UHFLIScopeReader(device_id)
scope_reader.initialize(configuration)
stimulus = UHFLIStimulus(device_id)

# Setup the UHFLIStimulus
demodulator = 1
demod_channel = 1
output_amplitude = 0.3
output_channel = 1
demodulation_parameter = 'R'  # 'phi', 'x' or 'y' also possible
stimulus.set_signal_output_amplitude(demod_channel, demodulator, output_amplitude)
stimulus.set_demodulation_enabled(demod_channel, True)
stimulus.set_signal_output_enabled(demod_channel, demodulator, True)
stimulus.set_output_enabled(output_channel, True)

# Setup Station and scanjob
station = Station()
station.gates = FakeGates()
scanjob = qtt.measurements.scans.scanjob_t()
scanjob.add_sweep(param=stimulus.set_oscillator_frequency(demod_channel), start=150e6, end=195e6, step=.2e6)
scanjob.add_minstrument(scope_reader.acquire_single_sample(demod_channel, demodulation_parameter, partial=True))

# Scan and plot
ds = qtt.measurements.scans.scan1D(station, scanjob)
set_points_name = 'oscillator{}_freq'.format(demod_channel)
data_points_name = 'demod{}_{}'.format(demod_channel, demodulation_parameter)
plt.plot(ds.arrays[set_points_name], ds.arrays[data_points_name])
plt.show()

# Disable channels and outputs
stimulus.set_demodulation_enabled(demod_channel, False)
stimulus.set_signal_output_enabled(demod_channel, demodulator, False)
stimulus.set_output_enabled(output_channel, False)
