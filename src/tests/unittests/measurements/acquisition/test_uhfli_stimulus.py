import unittest

from unittest.mock import patch, MagicMock, call
from qilib.utils import PythonJsonStructure

from qtt.measurements.acquisition import UHFLIStimulus


class TestLockInStimulus(unittest.TestCase):
    def setUp(self):
        self.adapter_mock = MagicMock()
        self.uhfli_mock = MagicMock()
        self.adapter_mock.instrument = self.uhfli_mock
        with patch('qtt.measurements.acquisition.uhfli_stimulus.InstrumentAdapterFactory') as factory_mock:
            factory_mock.get_instrument_adapter.return_value = self.adapter_mock
            self.uhfli_stimulus = UHFLIStimulus('mock42')

    def test_initialize(self):
        config = PythonJsonStructure(bla='blu')
        self.uhfli_stimulus.initialize(config)
        self.adapter_mock.apply.assert_called_once_with(config)

    def test_set_demodulation_enabled(self):
        getitem_mock = MagicMock()
        self.uhfli_mock.parameters.__getitem__ = getitem_mock

        self.uhfli_stimulus.set_demodulation_enabled(2, True)
        expected_calls = [call('demod2_streaming'), call()('ON')]
        getitem_mock.assert_has_calls(expected_calls)

        self.uhfli_stimulus.set_demodulation_enabled(1, False)
        expected_calls = [call('demod1_streaming'), call()('OFF')]
        getitem_mock.assert_has_calls(expected_calls)

    def test_set_output_enabled(self):
        getitem_mock = MagicMock()
        self.uhfli_mock.parameters.__getitem__ = getitem_mock

        self.uhfli_stimulus.set_output_enabled(2, True)
        expected_calls = [call('signal_output2_on'), call()('ON')]
        getitem_mock.assert_has_calls(expected_calls)

        self.uhfli_stimulus.set_output_enabled(1, False)
        expected_calls = [call('signal_output1_on'), call()('OFF')]
        getitem_mock.assert_has_calls(expected_calls)

    def test_set_oscillator_frequency(self):
        getitem_mock = MagicMock()
        self.uhfli_mock.parameters.__getitem__ = getitem_mock

        self.uhfli_stimulus.set_oscillator_frequency(5, 42.0)
        expected_calls = [call('oscillator5_freq'), call()(42.0)]
        getitem_mock.assert_has_calls(expected_calls)

    def test_set_oscillator_frequency_partial(self):
        getitem_mock = MagicMock()
        getitem_mock.return_value = 'FakeParameter'
        self.uhfli_mock.parameters.__getitem__ = getitem_mock

        parameter = self.uhfli_stimulus.set_oscillator_frequency(5)
        expected_calls = [call('oscillator5_freq')]
        getitem_mock.assert_has_calls(expected_calls)
        self.assertEqual(parameter, 'FakeParameter')

    def test_set_signal_output_enabled(self):
        getitem_mock = MagicMock()
        self.uhfli_mock.parameters.__getitem__ = getitem_mock

        self.uhfli_stimulus.set_signal_output_enabled(2, 3, True)
        expected_calls = [call('signal_output2_enable3'), call()(True)]
        getitem_mock.assert_has_calls(expected_calls)

    def test_set_signal_output_amplitude(self):
        getitem_mock = MagicMock()
        self.uhfli_mock.parameters.__getitem__ = getitem_mock

        self.uhfli_stimulus.set_signal_output_amplitude(1, 7, 0.42)
        expected_calls = [call('signal_output1_amplitude7'), call()(0.42)]
        getitem_mock.assert_has_calls(expected_calls)

    def test_demodulator_signal_input(self):
        getitem_mock = MagicMock()
        self.uhfli_mock.parameters.__getitem__ = getitem_mock

        self.uhfli_stimulus.set_demodulator_signal_input(8, 1)
        expected_calls = [call('demod8_signalin'), call()('Sig In 1')]
        getitem_mock.assert_has_calls(expected_calls)

        self.assertRaisesRegex(NotImplementedError, 'The input channel can *',
                               self.uhfli_stimulus.set_demodulator_signal_input, 8, 3)

    def test_connect_oscillator_to_demodulator(self):
        getitem_mock = MagicMock()
        self.uhfli_mock.parameters.__getitem__ = getitem_mock
        self.uhfli_stimulus.connect_oscillator_to_demodulator(3, 8)

        expected_calls = [call('demod8_oscillator'), call()(3)]
        getitem_mock.assert_has_calls(expected_calls)
