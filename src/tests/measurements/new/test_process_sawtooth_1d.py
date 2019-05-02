from unittest import TestCase
import numpy as np
from qilib.data_set import DataArray, DataSet
from scipy.signal import sawtooth

from qtt.measurements.new.process_sawtooth_1d import ProcessSawtooth1D


class TestProcessSawtooth1D(TestCase):

    def test_run_process_has_correct_shape(self):
        sample_rate = 21e7
        width = 0.9375
        resolution = 64
        period = resolution / sample_rate

        sawtooth_2d_processor = ProcessSawtooth1D()
        data_set = self.__dummy_data_set(period, sample_rate, width, resolution)
        output_result = sawtooth_2d_processor.run_process(data_set)

        image_shape = np.multiply(resolution, width)
        data_array = next(iter(output_result.data_arrays.values()))
        data_shape = data_array.T.shape
        self.assertEqual(image_shape, data_shape)

    @staticmethod
    def __dummy_time_data(period, sample_rate):
        return np.linspace(0, period, np.rint(period * sample_rate))

    @staticmethod
    def __dummy_scope_data(time_data, sawteeth_count, period, width):
        return sawtooth(2 * np.pi * sawteeth_count * time_data / period, width)

    @staticmethod
    def __dummy_data_array(set_array: DataArray, scope_data: np.ndarray, channel_index: int = 1, trace_number: int = 1):
        idenifier = 'ScopeTrace_{:03d}'.format(trace_number)
        label = 'Channel_{}'.format(channel_index)
        return DataArray(idenifier, label, preset_data=scope_data, set_arrays=[set_array])

    @staticmethod
    def __dummy_data_set(period, sample_rate, width, resolution):
        time_data = TestProcessSawtooth1D.__dummy_time_data(period, sample_rate)
        set_array = DataArray('ScopeTime', 'Time', unit='seconds', is_setpoint=True, preset_data=time_data)

        scope_data_1 = TestProcessSawtooth1D.__dummy_scope_data(time_data, resolution, period, width)
        data_array_1 = TestProcessSawtooth1D.__dummy_data_array(set_array, scope_data_1, channel_index=1,
                                                                trace_number=1)

        data_set = DataSet()
        data_set.user_data = {'resolution': resolution, 'width': width}
        data_set.add_array(data_array_1)

        return data_set
