import unittest

from mock import patch, MagicMock, call
from qilib.utils import PythonJsonStructure

from qtt.measurements.acquisition import LockInStimulus


class TestLockInStimulus(unittest.TestCase):
    def setUp(self):
        self.adapter_mock = MagicMock()
        self.uhfli_mock = MagicMock()
        self.adapter_mock.instrument = self.uhfli_mock
        with patch('qtt.measurements.acquisition.lock_in_stimulus.InstrumentAdapterFactory') as factory_mock:
            factory_mock.get_instrument_adapter.return_value = self.adapter_mock
            self.lock_in_stimulus = LockInStimulus('mock42')

    def test_initialize(self):
        config = PythonJsonStructure(bla='blu')
        self.lock_in_stimulus.initialize(config)
        self.adapter_mock.apply.assert_called_once_with(config)

    def test_set_demodulation_enabled(self):
        yamock = MagicMock()
        self.uhfli_mock.parameters.__getitem__ = yamock

        self.lock_in_stimulus.set_demodulation_enabled(2, True)
        expected_calls = [call('demod2_streaming'), call()('ON')]
        yamock.assert_has_calls(expected_calls)

        self.lock_in_stimulus.set_demodulation_enabled(1, False)
        expected_calls = [call('demod1_streaming'), call()('OFF')]
        yamock.assert_has_calls(expected_calls)

    def test_set_output_enabled(self):
        yamock = MagicMock()
        self.uhfli_mock.parameters.__getitem__ = yamock

        self.lock_in_stimulus.set_output_enabled(2, True)
        expected_calls = [call('signal_output2_on'), call()('ON')]
        yamock.assert_has_calls(expected_calls)

        self.lock_in_stimulus.set_output_enabled(1, False)
        expected_calls = [call('signal_output1_on'), call()('OFF')]
        yamock.assert_has_calls(expected_calls)

    def test_set_oscillator_frequency(self):
        yamock = MagicMock()
        self.uhfli_mock.parameters.__getitem__ = yamock

        self.lock_in_stimulus.set_oscillator_frequency(5, 42.0)
        expected_calls = [call('oscillator5_freq'), call()(42.0)]
        yamock.assert_has_calls(expected_calls)

    def test_set_signal_output_enabled(self):
        yamock = MagicMock()
        self.uhfli_mock.parameters.__getitem__ = yamock

        self.lock_in_stimulus.set_signal_output_enabled(3, 2, True)
        expected_calls = [call('signal_output2_enable3'), call()(True)]
        yamock.assert_has_calls(expected_calls)

    def test_set_signal_output_amplitude(self):
        yamock = MagicMock()
        self.uhfli_mock.parameters.__getitem__ = yamock

        self.lock_in_stimulus.set_signal_output_amplitude(7, 1, 0.42)
        expected_calls = [call('signal_output1_amplitude7'), call()(0.42)]
        yamock.assert_has_calls(expected_calls)
