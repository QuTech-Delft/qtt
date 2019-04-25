import numpy as np
from unittest import TestCase
from unittest.mock import patch, call

from qilib.data_set import DataSet
from qilib.utils import PythonJsonStructure

from qtt.measurements.new import UhfliScopeReader

class TestUhfliScopeReader(TestCase):

    @staticmethod
    def __patch_scope_reader(address):
        patch_object = 'qtt.measurements.new.uhfli_scope_reader.InstrumentAdapterFactory'
        with patch(patch_object) as factory_mock:
            scope_mock = UhfliScopeReader(address)
        return scope_mock, factory_mock

    def test_initialize_applies_configuration(self):
        mock_address = 'dev2331'
        mock_config = PythonJsonStructure(a=1, b='2', c=3.1415)
        scope_mock, factory_mock = TestUhfliScopeReader.__patch_scope_reader(mock_address)
        scope_mock.initialize(mock_config)

        factory_mock.get_instrument_adapter.assert_called_once_with('ZIUHFLIInstrumentAdapter', mock_address)
        scope_mock.adapter.apply.assert_called_once_with(mock_config)

    def test_prepare_acquisition_set_settings_correctly(self):
        mock_address = 'dev2332'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)
        scope_mock.prepare_acquisition()

        scope_mock.adapter.instrument.scope.set.assert_called_once_with('scopeModule/mode', 1)
        scope_mock.adapter.instrument.scope.subscribe.assert_called_once_with('/{}/scopes/0/wave'.format(mock_address))
        scope_mock.adapter.instrument.daq.sync.assert_called()
        scope_mock.adapter.instrument.daq.setInt.assert_called_once_with('/{}/scopes/0/enable'.format(mock_address), 1)

    def test_acquire_has_added_output_to_dataset(self):
        mock_address = 'dev2333'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)

        scope_length = 1000
        name = 'Test2'
        unit = 'MegaVolt'

        scope_mock.adapter.instrument.scope_length.return_value = scope_length
        scope_mock.adapter.instrument.scope_duration.get.return_value = 0.1
        scope_mock.adapter.instrument.scope_samplingrate_float.get.return_value = 115e3

        scope_mock.adapter.instrument.scope.getInt.return_value = 1
        scope_mock.adapter.instrument.scope.progress.return_value = [1]

        wave_nodepath = '/{0}/scopes/0/wave'.format(mock_address)
        scope_data = [[1, 2, 3 ,4], np.ndarray([1, 2, 3, 4, 5])]
        trace = [[{'wave': scope_data}], 0, 0]
        scope_mock.adapter.instrument.scope.read.return_value = {wave_nodepath: trace}
        scope_mock.adapter.instrument.Scope.names = ['Test1', name]
        scope_mock.adapter.instrument.Scope.units = ['GigaVolt', unit]

        data_set = DataSet()
        scope_mock.acquire(data_set)

        self.assertEqual(len(data_set.ScopeTime), scope_length)
        self.assertEqual(data_set.ScopeTime.label, 'Time')
        self.assertEqual(data_set.ScopeTime.unit, 'seconds')

        np.testing.assert_array_equal(data_set.ScopeTrace_001, scope_data[1])
        self.assertEqual(data_set.ScopeTrace_001.label, name)
        self.assertEqual(data_set.ScopeTrace_001.unit, unit)

    def test_acquire_raises_timeout(self):
        mock_address = 'dev2333a'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)

        scope_mock.adapter.instrument.scope_length.return_value = 1000
        scope_mock.adapter.instrument.scope_duration.get.return_value = 0.1
        scope_mock.adapter.instrument.scope_samplingrate_float.get.return_value = 115e3

        scope_mock.adapter.instrument.scope.getInt.return_value = 0
        scope_mock.adapter.instrument.scope.progress.return_value = [0.5]

        data_set = DataSet()
        with self.assertRaisesRegex(TimeoutError, 'Got 0 records after 0.0001 sec'):
            scope_mock.acquire(data_set, timeout=1e-4)

    def test_finalize_acquisition_set_settings_correctly(self):
        mock_address = 'dev2334'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)
        scope_mock.finalize_acquisition()

        command_text = '/{}/scopes/0/enable'.format(mock_address)
        scope_mock.adapter.instrument.daq.setInt.assert_called_once_with(command_text, 0)
        scope_mock.adapter.instrument.scope.finish.assert_called()

    def test_number_of_averages_getter(self):
        mock_address = 'dev2335'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)

        number_of_averages = 111
        scope_mock.adapter.instrument.scope_average_weight.get.return_value = number_of_averages
        self.assertEqual(number_of_averages, scope_mock.number_of_averages)
        scope_mock.adapter.instrument.scope_average_weight.get.assert_called_once()

    def test_number_of_averages_setter(self):
        mock_address = 'dev2336'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)

        number_of_averages = 222
        scope_mock.number_of_averages = number_of_averages
        scope_mock.adapter.instrument.scope_segments.set.assert_called_once_with('ON')
        scope_mock.adapter.instrument.scope_average_weight.set.assert_called_once_with(number_of_averages)

        number_of_averages = 1
        scope_mock.number_of_averages = number_of_averages
        scope_mock.adapter.instrument.scope_segments.set.assert_called_with('OFF')
        scope_mock.adapter.instrument.scope_average_weight.set.assert_called_with(number_of_averages)

    def test_input_range_getter(self):
        mock_address = 'dev2337'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)

        input_range = [1.25, 1.75]
        scope_mock.adapter.instrument.signal_input1_range.get.return_value = input_range[0]
        scope_mock.adapter.instrument.signal_input2_range.get.return_value = input_range[1]
        self.assertEqual(input_range, scope_mock.input_range)
        scope_mock.adapter.instrument.signal_input1_range.get.assert_called_once()
        scope_mock.adapter.instrument.signal_input2_range.get.assert_called_once()

    def test_input_range_setter(self):
        mock_address = 'dev2337'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)

        input_range = [0.75, 1.50]
        scope_mock.input_range = input_range
        scope_mock.adapter.instrument.signal_input1_range.set.assert_called_once_with(input_range[0])
        scope_mock.adapter.instrument.signal_input2_range.set.assert_called_once_with(input_range[1])

    def test_sample_rate_getter(self):
        mock_address = 'dev2338'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)

        sample_rate = 1.2e9
        scope_mock.adapter.instrument.scope_samplingrate_float.get.return_value = sample_rate
        self.assertEqual(sample_rate, scope_mock.sample_rate)
        scope_mock.adapter.instrument.scope_samplingrate_float.get.assert_called_once()

    def test_sample_setter(self):
        mock_address = 'dev2339'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)

        sample_rate = 113e7
        scope_mock.sample_rate = sample_rate
        scope_mock.adapter.instrument.scope_samplingrate_float.set.assert_called_once_with(sample_rate)

    def test_period_getter(self):
        mock_address = 'dev2340'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)

        period = 0.1234
        scope_mock.adapter.instrument.scope_duration.get.return_value = period
        self.assertEqual(period, scope_mock.period)
        scope_mock.adapter.instrument.scope_duration.get.assert_called_once()

    def test_period_setter(self):
        mock_address = 'dev2341'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)

        period = 0.4321
        scope_mock.period = period
        scope_mock.adapter.instrument.scope_duration.set.assert_called_once_with(period)

    def test_trigger_enabled_getter(self):
        mock_address = 'dev2342'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)

        is_enabled = True
        scope_mock.adapter.instrument.scope_trig_enable.get.return_value = 'ON'
        self.assertEqual(is_enabled, scope_mock.trigger_enabled)

        is_enabled = False
        scope_mock.adapter.instrument.scope_trig_enable.get.return_value = 'OFF'
        self.assertEqual(is_enabled, scope_mock.trigger_enabled)

        scope_mock.adapter.instrument.scope_trig_enable.get.return_value = 'MAYBE'
        error_text = "Unknown trigger value"
        with self.assertRaisesRegex(ValueError, error_text):
            scope_mock.trigger_enabled

    def test_trigger_enabled_setter(self):
        mock_address = 'dev2343'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)

        scope_mock.trigger_enabled = True
        scope_mock.adapter.instrument.scope_trig_enable.set.assert_called_once_with('ON')

        scope_mock.trigger_enabled = False
        scope_mock.adapter.instrument.scope_trig_enable.set.assert_called_with('OFF')

    def test_set_trigger_settings_has_correct_values(self):
        mock_address = 'dev2343'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)

        channel = 'Trig In 1'
        level = 0.2
        slope = 'Rise'
        delay = 0.1
        scope_mock.set_trigger_settings(channel, level, slope, delay)

        scope_mock.adapter.instrument.scope_trig_signal.set.assert_called_once_with(channel)
        scope_mock.adapter.instrument.scope_trig_level.set.assert_called_once_with(level)
        scope_mock.adapter.instrument.scope_trig_slope.set.assert_called_once_with(slope)
        scope_mock.adapter.instrument.scope_trig_delay.set.assert_called_once_with(delay)

    def test_enable_channels_getter(self):
        mock_address = 'dev2344'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)

        enabled_channels = 3
        scope_mock.adapter.instrument.scope_channels.get.return_value = enabled_channels
        self.assertEqual(scope_mock.enabled_channels, [1, 2])

        enabled_channels = 2
        scope_mock.adapter.instrument.scope_channels.get.return_value = enabled_channels
        self.assertEqual(scope_mock.enabled_channels, [2])

        enabled_channels = 1
        scope_mock.adapter.instrument.scope_channels.get.return_value = enabled_channels
        self.assertEqual(scope_mock.enabled_channels, [1])

    def test_enabled_channels_setter(self):
        mock_address = 'dev2345'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)

        enabled_channels = [1, 2]
        scope_mock.enabled_channels = enabled_channels
        scope_mock.adapter.instrument.scope_channels.set.assert_called_once_with(3)

        enabled_channels = [1]
        scope_mock.enabled_channels = enabled_channels
        scope_mock.adapter.instrument.scope_channels.set.assert_called_with(1)

        enabled_channels = [2]
        scope_mock.enabled_channels = enabled_channels
        scope_mock.adapter.instrument.scope_channels.set.assert_called_with(2)

        enabled_channels = [1, 2, 3]
        with self.assertRaisesRegex(ValueError, 'Invalid enabled channels specification'):
            scope_mock.enabled_channels = enabled_channels

    def test_input_signal_is_correctly_set(self):
        mock_address = 'dev2346'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)

        channel = 1
        attribute = 'Signal Input 1'

        scope_mock.set_input_signal(channel, attribute)
        scope_mock.adapter.instrument.scope_channel1_input.set.assert_called_once_with(attribute)
