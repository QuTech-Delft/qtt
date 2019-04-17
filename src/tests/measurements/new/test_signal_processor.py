from unittest import TestCase

from mock import patch
from numpy import array_equal
from numpy.ma import array
from qilib.data_set import DataSet, DataArray

from qtt.measurements.new.interfaces.signal_processor_interface import SignalProcessorInterface
from qtt.measurements.new.signal_processor import SignalProcessor


class TestSignalProcessor(TestCase):
    def test_run_processes(self):
        with patch('qtt.measurements.new.interfaces.signal_processor_interface.SignalProcessorInterface') as spi:
            class DummySignalProcessor(spi):
                def run_process(self, signal_data: DataSet) -> DataSet:
                    return signal_data

        signal_processor = SignalProcessor()
        signal_processor.add_signal_processor(DummySignalProcessor())

        data_set = DataSet()
        signal_processor.run_processes(data_set)
        spi.run_process.assert_called_once()
        spi.run_process.assert_called_with(data_set)

    def test_add_signal_processor(self):
        class DummySignalProcessor(SignalProcessorInterface):
            def run_process(self, signal_data: DataSet) -> DataSet:
                return signal_data

        signal_processor = SignalProcessor()
        signal_processor.add_signal_processor(DummySignalProcessor())
        self.assertEqual(len(signal_processor._signal_processors), 1)
        self.assertIsInstance(signal_processor._signal_processors[0], DummySignalProcessor)

    def test_run_process_single_signal_processor(self):
        data_set = DataSet(data_arrays=DataArray('x', 'x', preset_data=array([1, 2, 3, 4, 5])))

        class PlusOneSignalProcessor(SignalProcessorInterface):
            def run_process(self, signal_data: DataSet) -> DataSet:
                signal_data.data_arrays['x'] += 1
                return signal_data

        new_data_set = PlusOneSignalProcessor().run_process(data_set)

        self.assertIs(data_set.data_arrays['x'], new_data_set.data_arrays['x'])
        self.assertTrue(array_equal(new_data_set.data_arrays['x'], array([2, 3, 4, 5, 6])))
