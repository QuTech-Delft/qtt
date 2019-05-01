import numpy as np
from qilib.data_set import DataSet

from qtt.measurements.post_processing import SignalProcessorInterface



class ProcessSawtooth2D(SignalProcessorInterface):

    def run_process(self, data_set: DataSet)-> DataSet:
        """ Extract a 2D image from a double sawtooth signal."""

        width_x, width_y = data_set.user_data['width']
        resolution_x, resolution_y = data_set.user_data['resolution']

        processed_data = []
        for _, data_array in data_set.data_arrays.items():
            ProcessSawtooth2D._check_samples_sawtooth_x(data_array, width_x, resolution_y)
            ProcessSawtooth2D._check_samples_sawtooth_y(data_array, width_y)

            sample_count = len(data_array)
            samples_sawtooth_x = int(sample_count / resolution_y)
            samples_edge_x = int(sample_count / resolution_y * width_x)

            samples_egde_y = int(sample_count * width_y)
            offsets = np.arange(0, samples_egde_y, samples_sawtooth_x, dtype=np.int)

            sliced_data = [data_array[o:o + samples_edge_x] for o in offsets]
            processed_data.append(np.array(sliced_data))

        return processed_data

    @staticmethod
    def _check_samples_sawtooth_y(data_array, width_y):
        samples_rising_edge = len(data_array) * width_y
        if not ProcessSawtooth2D.__is_integer(samples_rising_edge):
            raise AssertionError('Invalid rising edge Y (samples {})'.format(samples_rising_edge))

    @staticmethod
    def _check_samples_sawtooth_x(data_array, width_x, resolution_y):
        samples_rising_edge = len(data_array) / resolution_y * width_x
        if not ProcessSawtooth2D.__is_integer(samples_rising_edge):
            raise AssertionError('Invalid rising edge X (samples {})'.format(samples_rising_edge))

    @staticmethod
    def _check_resolution_sawtooth_x(data_array, width_x, resolution_x):
        pass

    @staticmethod
    def __is_integer(value):
        return value % 1 == 0
