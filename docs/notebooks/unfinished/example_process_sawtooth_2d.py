from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from qilib.data_set import DataArray, DataSet
from scipy.signal import sawtooth

from qtt.measurements.post_processing import SignalProcessorRunner, ProcessSawtooth2D


# DUMMY DATASET CREATION

pixels_x = 100
upwards_edge_pixels_x = 90

pixels_y = 50
upwards_edge_pixels_y = 45

sample_rate = 21e7  # samples per seconds.
period = pixels_x * pixels_y / sample_rate  # seconds

time = np.linspace(0, period, np.rint(period * sample_rate))
set_array = DataArray('ScopeTime', 'Time', unit='seconds', is_setpoint=True, preset_data=time)

width = [upwards_edge_pixels_x/pixels_x, upwards_edge_pixels_y/pixels_y]
resolution = [pixels_x, pixels_y]


def create_dummy_data_array(width: float, sawteeth_count: int, channel_index: int = 1, trace_number: int = 1):
    identifier = 'ScopeTrace_{:03d}'.format(trace_number)
    label = 'Channel_{}'.format(channel_index)
    scope_data = sawtooth(2 * np.pi * sawteeth_count * time / period, width)
    return DataArray(identifier, label, preset_data=scope_data, set_arrays=[set_array])


data_set = DataSet()
data_set.user_data = {'resolution': resolution, 'width': width}

data_array_x = create_dummy_data_array(width=width[0], sawteeth_count=1, channel_index=1, trace_number=1)
data_set.add_array(data_array_x)

data_array_y = create_dummy_data_array(width=width[1], sawteeth_count=resolution[1], channel_index=2, trace_number=2)
data_set.add_array(data_array_y)


## EXAMPLE PRCCESSING 2D SAWTOOTH

signal_processor = SignalProcessorRunner()
signal_processor.add_signal_processor(ProcessSawtooth2D())
processed_data_set = signal_processor.run(data_set)


## PLOTTING

color_cycler = cycle('bgrcmk')

def plot_1D_dataset(data_set, label_x, label_y, figure_number=100):
    plt.figure(figure_number)
    plt.clf()
    plt.xlabel(label_x)
    plt.ylabel(label_y)

    for name, trace_data in data_set.data_arrays.items():
        plt.plot(trace_data.set_arrays[0].flatten(), trace_data.flatten(),
                 color=next(color_cycler), label=name)

    plt.legend(loc='upper left')
    plt.show()


def plot_2d_result(processed_2d_data, figure_number=100):
    plt.figure(figure_number)
    plt.clf()
    plt.imshow(processed_2d_data)
    plt.show()


plot_1D_dataset(data_set, 'Time', 'Dummy')
[plot_2d_result(data_array) for data_array in processed_data_set.data_arrays.values()]
