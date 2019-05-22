import numpy as np
from typing import Tuple
from scipy.signal import sawtooth

from unittest import TestCase
from qilib.data_set import DataArray, DataSet
from qtt.measurements.post_processing import ProcessSawtooth2D


class TestProcessSawtooth2D(TestCase):

    def test_invalid_sample_count_slow_sawtooth(self):
        sample_rate = 21e7
        width = [0.9, 0.9]
        resolution = [101, 50]
        period = resolution[0] * resolution[1] / sample_rate
        processing = 'center'

        sawtooth_2d_processor = ProcessSawtooth2D()
        data_set = TestProcessSawtooth2D.__dummy_data_set(period, sample_rate, width, resolution, processing)
        self.assertRaisesRegex(AssertionError, 'Invalid rising edge X *', sawtooth_2d_processor.run_process, data_set)

    def test_invalid_sample_count_fast_sawtooth(self):
        sample_rate = 21e7
        width = [0.9, 0.91234]
        resolution = [100, 50]
        period = resolution[0] * resolution[1] / sample_rate
        processing = 'center'

        sawtooth_2d_processor = ProcessSawtooth2D()
        data_set = TestProcessSawtooth2D.__dummy_data_set(period, sample_rate, width, resolution, processing)
        self.assertRaisesRegex(AssertionError, 'Invalid rising edge Y *', sawtooth_2d_processor.run_process, data_set)

    def test_check_matching_cuttoff(self):
        sample_rate = 21e7
        width = [0.984375, 0.9375]
        resolution = [64, 32]
        period = resolution[0] * resolution[1] / sample_rate
        processing = 'center'

        sawtooth_2d_processor = ProcessSawtooth2D()
        data_set = TestProcessSawtooth2D.__dummy_data_set(period, sample_rate, width, resolution, processing)
        self.assertRaisesRegex(AssertionError, 'Pixel ratio is incompatible with cuttoff*', sawtooth_2d_processor.run_process, data_set)

    def test_run_process_left_has_correct_shape(self):
        sample_rate = 21e7
        width = [0.9375, 0.9375]
        resolution = [64, 32]
        period = resolution[0] * resolution[1] / sample_rate
        processing = 'left'

        sawtooth_2d_processor = ProcessSawtooth2D()
        data_set = TestProcessSawtooth2D.__dummy_data_set(period, sample_rate, width, resolution, processing)
        output_result = sawtooth_2d_processor.run_process(data_set)
        for item in output_result.data_arrays.values():
            image_shape = list(np.multiply(resolution, width))
            data_shape = list(item.T.shape)
            self.assertEqual(image_shape, data_shape)

    def test_run_process_center_has_correct_shape(self):
        sample_rate = 21e7
        width = [0.9375, 0.9375]
        resolution = [64, 32]
        period = resolution[0] * resolution[1] / sample_rate
        processing = 'center'

        sawtooth_2d_processor = ProcessSawtooth2D()
        data_set = TestProcessSawtooth2D.__dummy_data_set(period, sample_rate, width, resolution, processing)
        output_result = sawtooth_2d_processor.run_process(data_set)
        for item in output_result.data_arrays.values():
            image_shape = list(np.multiply(resolution, width))
            data_shape = list(item.T.shape)
            self.assertEqual(image_shape, data_shape)

    def test_run_process_right_has_correct_shape(self):
        sample_rate = 21e7
        width = [0.9375, 0.9375]
        resolution = [64, 32]
        period = resolution[0] * resolution[1] / sample_rate
        processing = 'right'

        sawtooth_2d_processor = ProcessSawtooth2D()
        data_set = TestProcessSawtooth2D.__dummy_data_set(period, sample_rate, width, resolution, processing)
        output_result = sawtooth_2d_processor.run_process(data_set)
        for item in output_result.data_arrays.values():
            image_shape = list(np.multiply(resolution, width))
            data_shape = list(item.T.shape)
            self.assertEqual(image_shape, data_shape)

    @staticmethod
    def __dummy_time_data(period, sample_rate):
        return np.linspace(0, period, np.rint(period * sample_rate))

    @staticmethod
    def __dummy_scope_data(time_data, sawteeth_count, period, width):
        return sawtooth(2 * np.pi * sawteeth_count * time_data / period, width)

    @staticmethod
    def __dummy_data_array(set_array: DataArray, scope_data: np.ndarray, channel_index: int = 1, trace_number: int = 1) -> DataArray:
        idenifier = 'ScopeTrace_{:03d}'.format(trace_number)
        label = 'Channel_{}'.format(channel_index)
        return DataArray(idenifier, label, preset_data=scope_data, set_arrays=[set_array])

    @staticmethod
    def __dummy_data_set(period: float, sample_rate: float, width: Tuple[int, int], resolution: Tuple[int, int], processing: str) -> DataSet:
        time_data = TestProcessSawtooth2D.__dummy_time_data(period, sample_rate)
        set_array = DataArray('ScopeTime', 'Time', unit='seconds', is_setpoint=True, preset_data=time_data)

        scope_data_1 = TestProcessSawtooth2D.__dummy_scope_data(time_data, resolution[0], period, width[0])
        data_array_1 = TestProcessSawtooth2D.__dummy_data_array(set_array, scope_data_1, channel_index=1, trace_number=1)

        scope_data_2 = TestProcessSawtooth2D.__dummy_scope_data(time_data, resolution[1], period, width[1])
        data_array_2 = TestProcessSawtooth2D.__dummy_data_array(set_array, scope_data_2, channel_index=2, trace_number=2)

        data_set = DataSet()
        data_set.user_data = {'resolution': resolution, 'width': width, 'processing': processing}
        data_set.add_array(data_array_1)
        data_set.add_array(data_array_2)

        return data_set
