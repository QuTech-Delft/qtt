from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from qilib.data_set import DataArray, DataSet
from scipy.signal import sawtooth

from qtt.measurements.post_processing import SignalProcessorRunner, ProcessSawtooth2D


# DUMMY CREATION

period = 64 * 32 / 14e6  # seconds
sample_rate = 14e6  # samples per seconds.

time = np.linspace(0, period, np.rint(period * sample_rate))
set_array = DataArray('ScopeTime', 'Time', unit='seconds', is_setpoint=True, preset_data=time)

width = [62/64, 31/32]
resolution = [64, 32]


def create_dummy_data_array(width: float, sawteeth_count: int, channel_index: int = 1, trace_number: int = 1):
    idenifier = 'ScopeTrace_{:03d}'.format(trace_number)
    label = 'Channel_{}'.format(channel_index)
    scope_data = sawtooth(2 * np.pi * sawteeth_count * time / period, width)
    return DataArray(idenifier, label, preset_data=scope_data, set_arrays=[set_array])


data_set = DataSet()
data_set.user_data = {'resolution': resolution, 'width': width}

data_array_x = create_dummy_data_array(width=width[0], sawteeth_count=1, channel_index=1, trace_number=1)
data_set.add_array(data_array_x)

data_array_y = create_dummy_data_array(width=width[1], sawteeth_count=resolution[1], channel_index=2, trace_number=2)
data_set.add_array(data_array_y)


color_cycler = cycle('bgrcmk')

def plot_1D_dataset(data_set, label_x, label_y, figure_number=100):
    plt.figure(figure_number)
    plt.xlabel(label_x)
    plt.ylabel(label_y)

    for name, trace_data in data_set.data_arrays.items():
        plt.plot(trace_data.set_arrays[0].flatten(), trace_data.flatten(),
                 color=next(color_cycler), label=name)

    plt.legend(loc='upper right')
    plt.show()


plot_1D_dataset(data_set, 'Time', 'Dummy')

signal_processor = SignalProcessorRunner()
signal_processor.add_signal_processor(ProcessSawtooth2D())


processed_data_set = signal_processor.run(data_set)

plt.figure(100)
plt.clf()
plt.imshow(processed_data_set[0])
plt.show()

plt.figure(101)
plt.clf()
plt.imshow(processed_data_set[1])
plt.show()