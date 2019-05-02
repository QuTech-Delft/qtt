import numpy as np
from qilib.data_set import DataSet, DataArray

from qtt.measurements.post_processing import SignalProcessorInterface


class ProcessSawtooth2D(SignalProcessorInterface):

    def run_process(self, data_set: DataSet)-> DataSet:
        """ Extracts a 2D image from a readout dot responce measured with an acquisition device.

        Args:
            data_set: The readout dot reponse data coming from the acquisition device. The data
                      user data of the data set should contain the width and resolution settings.

        Returns:
            A data set which contains a 2D image with the charge stability diagram.
        """
        width_x, width_y = data_set.user_data['width']
        resolution_x, resolution_y = data_set.user_data['resolution']

        output_data_set = DataSet()
        output_data_set.user_data = data_set.user_data
        for _, data_array in data_set.data_arrays.items():
            ProcessSawtooth2D.__check_sample_count_slow_sawtooth(data_array, width_y)
            ProcessSawtooth2D.__check_sample_count_fast_sawtooth(data_array, width_x, resolution_x, resolution_y)
            ProcessSawtooth2D.__check_matching_cuttoff(width_x, width_y, resolution_x, resolution_y)

            sample_count = len(data_array)
            samples_sawtooth_x = int(sample_count / resolution_y)
            samples_edge_x = int(sample_count / resolution_y * width_x)

            samples_egde_y = int(sample_count * width_y)
            offsets = np.arange(0, samples_egde_y, samples_sawtooth_x, dtype=np.int)

            sliced_data = np.array([data_array[o:o + samples_edge_x] for o in offsets])
            identifier = '{0}_SawtoothProcessed2D'.format(data_array.name)
            result_data = DataArray(identifier, data_array.label, preset_data=sliced_data)
            output_data_set.add_array(result_data)

        return output_data_set

    @staticmethod
    def __check_sample_count_fast_sawtooth(data_array, width_x, resolution_x, resolution_y):
        expected_pixels_x = resolution_x * width_x
        samples_rising_edge = len(data_array) / resolution_y * width_x
        if not ProcessSawtooth2D.__is_integer(samples_rising_edge) or expected_pixels_x != samples_rising_edge:
            raise AssertionError('Invalid rising edge X (samples {} not an integer)!'.format(samples_rising_edge))

    @staticmethod
    def __check_sample_count_slow_sawtooth(data_array, width_y):
        samples_rising_edge = len(data_array) * width_y
        if not ProcessSawtooth2D.__is_integer(samples_rising_edge):
            raise AssertionError('Invalid rising edge Y (samples {} not an integer)!'.format(samples_rising_edge))

    @staticmethod
    def __check_matching_cuttoff(width_x, width_y, resolution_x, resolution_y):
        pixels_x = resolution_x * width_x
        pixels_y = resolution_y * width_y
        if ProcessSawtooth2D.__is_integer(pixels_x / pixels_y) or ProcessSawtooth2D.__is_integer(pixels_y / pixels_x):
            return
        error_message = 'Pixel ratio is incompatible with cuttoff (x={}, y={})!'.format(pixels_x, pixels_y)
        raise AssertionError(error_message)

    @staticmethod
    def __is_integer(value):
        return value % 1 == 0
