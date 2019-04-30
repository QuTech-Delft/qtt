import os
from itertools import cycle

import matplotlib.pyplot as plt
from qcodes import Station

from qtt.measurements.acquisition import UHFLIScopeReader, load_configuration


# FUNCTIONS FOR PLOTTING

color_cycler = cycle('bgrcmk')


def plot_1D_dataset(plot, records, label_x, label_y):
    plot.clf()
    plot.xlabel(label_x)
    plot.ylabel(label_y)
    for record in records:
        plot.plot(record.set_arrays[0].flatten(), record.flatten(), color=next(color_cycler), label=record.name)
        plot.legend(loc='upper right')
        plot.draw()
        plot.pause(0.001)


# CREATE THE SCOPE READER AND STATION

device_id = 'dev2338'
scope_reader = UHFLIScopeReader(device_id)
uhfli = scope_reader.adapter.instrument
station = Station(uhfli, update_snapshot=False)


# INITIALIZE THE SCOPE READER

file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uhfli.dat')
configuration = load_configuration(file_path)
scope_reader.initialize(configuration)


# PREPARE THE SCOPE FOR READOUT

scope_reader.number_of_averages = 1
scope_reader.input_range = (0.5, 1.0)
scope_reader.sample_rate = 450e6
scope_reader.period = 1e-5

scope_reader.enabled_channels = (1, )
scope_reader.set_input_signal(1, 'Signal Input 1')

scope_reader.trigger_enabled = False
scope_reader.trigger_channel = 'Trig Input 1'
scope_reader.trigger_level = 0.100
scope_reader.trigger_slope = 'Rise'
scope_reader.trigger_delay = 0

scope_reader.start_acquisition()



# READOUT THE SCOPE AND PLOT THE DATA

plt.figure()
plt.ion()
plt.show()

samples = 10

for number in range(1, samples):
    records = scope_reader.acquire(number_of_records=10)
    plot_1D_dataset(plt, records, 'Time [sec.]', 'Amplitude [V]')

scope_reader.stop_acquisition()
