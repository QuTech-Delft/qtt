import os
from itertools import cycle

import matplotlib.pyplot as plt
from qcodes import Measure, Station
from qcodes.instrument_drivers.ZI.ZIUHFLI import ZIUHFLI

from qilib.data_set import DataSet
from qtt.measurements.new import UhfliScopeReader
from qtt.measurements.new.configuration_storage import (load_configuration,
                                                        save_configuration)


# FUNCTIONS FOR PLOTTING

color_cycler = cycle('bgrcmk')


def plot_1D_dataset(plot, dataset, name_x, names_y, label_x, label_y):
    x_data = getattr(dataset, name_x)

    if isinstance(names_y, str):
        y_data_list = [getattr(dataset, names_y)]
    elif isinstance(names_y, list):
        y_data_list = [getattr(dataset, n) for n in names_y]
    else:
        raise ValueError('Invalid name_y_data argument! Must be list[str] or str!')

    plot.clf()
    for y_data in y_data_list:
        plot.plot(x_data.flatten(), y_data.flatten(), color=next(color_cycler), label=y_data.name)

    plot.xlabel(label_x)
    plot.ylabel(label_y)
    plot.legend(loc='upper right')
    plot.draw()
    plot.pause(0.001)


# CREATE THE QCODES STATION

device_id = 'dev2338'
uhfli = ZIUHFLI('uhfli', device_id)
station = Station(uhfli, update_snapshot=False)


# CREATE THE SCOPE READER

scope_reader = UhfliScopeReader(device_id)

file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uhfli.dat')
# save_configuration(file_path, scope_reader._adapter)

configuration = load_configuration(file_path)
scope_reader.initialize(configuration)


# PREPARE THE SCOPE FOR READOUT

scope_reader.number_of_averages = 1
scope_reader.input_range = [0.5, 1.0]
scope_reader.sample_rate = 450e6
scope_reader.period = 1e-5

scope_reader.enabled_channels = [1]
scope_reader.set_input_signal(1, 'Signal Input 1')
scope_reader.trigger_enabled = False

scope_reader.prepare_acquisition()


# READOUT THE SCOPE AND PLOT THE DATA

data_set = DataSet()
plt.figure()
plt.ion()
plt.show()

samples = 10
for number in range(1, samples):
    scope_reader.acquire(data_set)
    plot_1D_dataset(plt, data_set, 'ScopeTime', 'ScopeTrace_{:03d}'.format(number), 'Time [sec.]', 'Amplitude [V]')

scope_reader.finalize_acquisition()
