import numpy as np
from unittest import TestCase
from unittest.mock import patch

from qilib.utils import PythonJsonStructure

from qtt.measurements.acquisition import UHFLIScopeReader


class TestUhfliScopeReader(TestCase):

    @staticmethod
    def __patch_scope_reader(address):
        patch_object = 'qtt.measurements.acquisition.uhfli_scope_reader.InstrumentAdapterFactory'
        with patch(patch_object) as factory_mock:
            scope_mock = UHFLIScopeReader(address)
            scope_mock.adapter.address = address
        return scope_mock, factory_mock

    def test_initialize_applies_configuration(self):
        mock_address = 'dev2331'
        mock_config = PythonJsonStructure(a=1, b='2', c=3.1415)
        scope_mock, factory_mock = TestUhfliScopeReader.__patch_scope_reader(mock_address)
        self.assertEqual(mock_address, scope_mock.adapter.address)
        scope_mock.initialize(mock_config)

        factory_mock.get_instrument_adapter.assert_called_once_with('ZIUHFLIInstrumentAdapter', mock_address)
        scope_mock.adapter.apply.assert_called_once_with(mock_config)
        self.assertEqual(mock_address, scope_mock.adapter.address)

    def test_prepare_acquisition_set_settings_correctly(self):
        mock_address = 'dev2332'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)
        self.assertEqual(mock_address, scope_mock.adapter.address)
        scope_mock.start_acquisition()

        scope_mock.adapter.instrument.scope.set.assert_called_once_with('scopeModule/mode', 1)
        scope_mock.adapter.instrument.scope.subscribe.assert_called_once_with('/{}/scopes/0/wave'.format(mock_address))
        scope_mock.adapter.instrument.daq.sync.assert_called()
        scope_mock.adapter.instrument.daq.setInt.assert_called_once_with('/{}/scopes/0/enable'.format(mock_address), 1)

    def test_acquire_has_added_output_to_dataset(self):
        mock_address = 'dev2333'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)
        self.assertEqual(mock_address, scope_mock.adapter.address)

        scope_length = 1000
        name = 'Test2'
        unit = 'MegaVolt'

        scope_mock.adapter.instrument.scope_length.return_value = scope_length
        scope_mock.adapter.instrument.scope_duration.get.return_value = 0.1
        scope_mock.adapter.instrument.scope_samplingrate_float.get.return_value = 115e3

        scope_mock.adapter.instrument.scope.getInt.return_value = 1
        scope_mock.adapter.instrument.scope.progress.return_value = [1]

        wave_nodepath = '/{0}/scopes/0/wave'.format(mock_address)
        scope_data = np.array([np.random.rand(1000), np.random.rand(1000)])
        trace = [[{'wave': scope_data}], 0, 0]
        scope_mock.adapter.instrument.scope.read.return_value = {wave_nodepath: trace}
        scope_mock.adapter.instrument.Scope.names = ['Test1', name]
        scope_mock.adapter.instrument.Scope.units = ['GigaVolt', unit]

        scope_trace_001 = scope_mock.acquire()
        self.assertEqual(len(scope_trace_001[0].set_arrays[0]), scope_length)
        self.assertEqual(scope_trace_001[0].name, 'ScopeTrace_000')
        self.assertEqual(scope_trace_001[0].label, 'Channel_0')

    def test_acquire_raises_timeout(self):
        mock_address = 'dev2333a'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)
        self.assertEqual(mock_address, scope_mock.adapter.address)

        scope_mock.adapter.instrument.scope_length.return_value = 1000
        scope_mock.adapter.instrument.scope_duration.get.return_value = 0.1
        scope_mock.adapter.instrument.scope_samplingrate_float.get.return_value = 115e3

        scope_mock.adapter.instrument.scope.getInt.return_value = 0
        scope_mock.adapter.instrument.scope.progress.return_value = [0.5]

        with self.assertRaisesRegex(TimeoutError, 'Got 0 records after 0.0001 sec'):
            scope_mock.acquire(timeout=1e-4)

    def test_finalize_acquisition_set_settings_correctly(self):
        mock_address = 'dev2334'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)
        self.assertEqual(mock_address, scope_mock.adapter.address)
        scope_mock.stop_acquisition()

        command_text = '/{}/scopes/0/enable'.format(mock_address)
        scope_mock.adapter.instrument.daq.setInt.assert_called_once_with(command_text, 0)
        scope_mock.adapter.instrument.scope.finish.assert_called()

    def test_number_of_averages_getter(self):
        mock_address = 'dev2335'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)
        self.assertEqual(mock_address, scope_mock.adapter.address)

        number_of_averages = 111
        scope_mock.adapter.instrument.scope_average_weight.get.return_value = number_of_averages
        self.assertEqual(number_of_averages, scope_mock.number_of_averages)
        scope_mock.adapter.instrument.scope_average_weight.get.assert_called_once()

    def test_number_of_averages_setter(self):
        mock_address = 'dev2336'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)
        self.assertEqual(mock_address, scope_mock.adapter.address)

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
        self.assertEqual(mock_address, scope_mock.adapter.address)

        input_range = (1.25, 1.75)
        scope_mock.adapter.instrument.signal_input1_range.get.return_value = input_range[0]
        scope_mock.adapter.instrument.signal_input2_range.get.return_value = input_range[1]
        self.assertEqual(input_range, scope_mock.input_range)
        scope_mock.adapter.instrument.signal_input1_range.get.assert_called_once()
        scope_mock.adapter.instrument.signal_input2_range.get.assert_called_once()

    def test_input_range_setter(self):
        mock_address = 'dev2338'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)
        self.assertEqual(mock_address, scope_mock.adapter.address)

        input_range = (0.75, 1.50)
        scope_mock.input_range = input_range
        scope_mock.adapter.instrument.signal_input1_range.set.assert_called_once_with(input_range[0])
        scope_mock.adapter.instrument.signal_input2_range.set.assert_called_once_with(input_range[1])

    def test_sample_rate_getter(self):
        mock_address = 'dev2340'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)
        self.assertEqual(mock_address, scope_mock.adapter.address)

        sample_rate = 1.2e9
        scope_mock.adapter.instrument.scope_samplingrate_float.get.return_value = sample_rate
        self.assertEqual(sample_rate, scope_mock.sample_rate)
        scope_mock.adapter.instrument.scope_samplingrate_float.get.assert_called_once()

    def test_sample_setter(self):
        mock_address = 'dev2341'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)
        self.assertEqual(mock_address, scope_mock.adapter.address)

        period = 0.54321
        scope_mock.adapter.instrument.scope_duration.get.return_value = period

        sample_rate = 113e7
        scope_mock.sample_rate = sample_rate
        scope_mock.adapter.instrument.scope_duration.get.assert_called_once()
        scope_mock.adapter.instrument.scope_samplingrate_float.set.assert_called_once_with(sample_rate)
        scope_mock.adapter.instrument.scope_duration.set.assert_called_once_with(period)

    def test_get_nearest_sample_rate_gets_value_correctly(self):
        mock_address = 'dev2341'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)
        self.assertEqual(mock_address, scope_mock.adapter.address)

        actual_nearest_value = 149e3
        scope_mock.adapter.instrument.round_to_nearest_sampling_frequency.return_value = actual_nearest_value

        guess_value = 150e3
        neasest_value = scope_mock.get_nearest_sample_rate(guess_value)
        scope_mock.adapter.instrument.round_to_nearest_sampling_frequency.assert_called_once_with(guess_value)
        self.assertEqual(actual_nearest_value, neasest_value)

    def test_period_getter(self):
        mock_address = 'dev2342'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)
        self.assertEqual(mock_address, scope_mock.adapter.address)

        period = 0.1234
        scope_mock.adapter.instrument.scope_duration.get.return_value = period
        self.assertEqual(period, scope_mock.period)
        scope_mock.adapter.instrument.scope_duration.get.assert_called_once()

    def test_period_setter(self):
        mock_address = 'dev2343'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)
        self.assertEqual(mock_address, scope_mock.adapter.address)

        period = 0.4321
        scope_mock.period = period
        scope_mock.adapter.instrument.scope_duration.set.assert_called_once_with(period)

    def test_trigger_enabled_getter(self):
        mock_address = 'dev2344'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)
        self.assertEqual(mock_address, scope_mock.adapter.address)

        is_enabled = True
        scope_mock.adapter.instrument.scope_trig_enable.get.return_value = 'ON'
        self.assertEqual(is_enabled, scope_mock.trigger_enabled)

        is_enabled = False
        scope_mock.adapter.instrument.scope_trig_enable.get.return_value = 'OFF'
        self.assertEqual(is_enabled, scope_mock.trigger_enabled)

        scope_mock.adapter.instrument.scope_trig_enable.get.return_value = 'MAYBE'
        error_text = "Unknown trigger value"
        with self.assertRaisesRegex(ValueError, error_text):
            trigger_value = scope_mock.trigger_enabled
            self.assertIsNone(trigger_value)

    def test_trigger_enabled_setter(self):
        mock_address = 'dev2345'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)
        self.assertEqual(mock_address, scope_mock.adapter.address)

        scope_mock.trigger_enabled = True
        scope_mock.adapter.instrument.scope_trig_enable.set.assert_called_once_with('ON')

        scope_mock.trigger_enabled = False
        scope_mock.adapter.instrument.scope_trig_enable.set.assert_called_with('OFF')

    def test_set_trigger_settings_have_correct_values(self):
        mock_address = 'dev2346'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)
        self.assertEqual(mock_address, scope_mock.adapter.address)

        channel = 'Trig In 1'
        scope_mock.trigger_channel = channel
        scope_mock.adapter.instrument.scope_trig_signal.set.assert_called_once_with(channel)

        level = 0.2
        scope_mock.trigger_level = level
        scope_mock.adapter.instrument.scope_trig_level.set.assert_called_once_with(level)

        slope = 'Rise'
        scope_mock.trigger_slope = slope
        scope_mock.adapter.instrument.scope_trig_slope.set.assert_called_once_with(slope)

        delay = 0.1
        scope_mock.trigger_delay = delay
        scope_mock.adapter.instrument.scope_trig_delay.set.assert_called_once_with(delay)

    def test_get_trigger_settings_have_correct_values(self):
        mock_address = 'dev2347'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)
        self.assertEqual(mock_address, scope_mock.adapter.address)

        channel = 'Trig In 1'
        scope_mock.adapter.instrument.scope_trig_signal.get.return_value = channel
        self.assertEqual(channel, scope_mock.trigger_channel)

        level = 0.2
        scope_mock.adapter.instrument.scope_trig_level.get.return_value = level
        self.assertEqual(level, scope_mock.trigger_level)

        slope = 'Rise'
        scope_mock.adapter.instrument.scope_trig_slope.get.return_value = slope
        self.assertEqual(slope, scope_mock.trigger_slope)

        delay = 0.1
        scope_mock.adapter.instrument.scope_trig_delay.get.return_value = delay
        self.assertEqual(delay, scope_mock.trigger_delay)

    def test_enable_channels_getter(self):
        mock_address = 'dev2348'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)
        self.assertEqual(mock_address, scope_mock.adapter.address)

        enabled_channels = 3
        scope_mock.adapter.instrument.scope_channels.get.return_value = enabled_channels
        self.assertEqual(scope_mock.enabled_channels, (1, 2))

        enabled_channels = 2
        scope_mock.adapter.instrument.scope_channels.get.return_value = enabled_channels
        self.assertEqual(scope_mock.enabled_channels, (2, ))

        enabled_channels = 1
        scope_mock.adapter.instrument.scope_channels.get.return_value = enabled_channels
        self.assertEqual(scope_mock.enabled_channels, (1, ))

    def test_enabled_channels_setter(self):
        mock_address = 'dev2349'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)
        self.assertEqual(mock_address, scope_mock.adapter.address)

        enabled_channels = (1, 2)
        scope_mock.enabled_channels = enabled_channels
        scope_mock.adapter.instrument.scope_channels.set.assert_called_once_with(3)

        enabled_channels = (1, )
        scope_mock.enabled_channels = enabled_channels
        scope_mock.adapter.instrument.scope_channels.set.assert_called_with(1)

        enabled_channels = (2, )
        scope_mock.enabled_channels = enabled_channels
        scope_mock.adapter.instrument.scope_channels.set.assert_called_with(2)

    def test_input_signal_is_correctly_set(self):
        mock_address = 'dev2350'
        scope_mock, _ = TestUhfliScopeReader.__patch_scope_reader(mock_address)
        self.assertEqual(mock_address, scope_mock.adapter.address)

        channel = 1
        attribute = 'Signal Input 1'

        scope_mock.set_input_signal(channel, attribute)
        scope_mock.adapter.instrument.scope_channel1_input.set.assert_called_once_with(attribute)
