from unittest import TestCase

from mock import patch
from numpy import array_equal
from numpy.ma import array
from qilib.data_set import DataSet, DataArray

from qtt.measurements.new.interfaces.signal_processor_interface import SignalProcessorInterface
from qtt.measurements.new.signal_processor_runner import SignalProcessorRunner


class TestSignalProcessorRunner(TestCase):
    @staticmethod
    def test_run():
        with patch('qtt.measurements.new.interfaces.signal_processor_interface.SignalProcessorInterface') as spi:
            class DummySignalProcessor(spi):
                def __init__(self):
                    self._signal_data = None

                def run_process(self, signal_data: DataSet) -> DataSet:
                    self._signal_data = signal_data
                    return signal_data

        signal_processor_runner = SignalProcessorRunner()
        signal_processor_runner.add_signal_processor(DummySignalProcessor())

        data_set = DataSet()
        signal_processor_runner.run(data_set)
        spi.run_process.assert_called_once()
        spi.run_process.assert_called_with(data_set)

    def test_add_signal_processor(self):
        class DummySignalProcessor(SignalProcessorInterface):
            def __init__(self):
                self._signal_data = None

            def run_process(self, signal_data: DataSet) -> DataSet:
                self._signal_data = signal_data
                return signal_data

        signal_processor = SignalProcessorRunner()
        signal_processor.add_signal_processor(DummySignalProcessor())
        self.assertEqual(len(signal_processor._signal_processors), 1)
        self.assertIsInstance(signal_processor._signal_processors[0], DummySignalProcessor)

    def test_run_process_without_signal_processor(self):
        data_set = DataSet(data_arrays=DataArray('x', 'x', preset_data=array([1, 2, 3, 4, 5])))

        signal_processor_runner = SignalProcessorRunner()

        new_data_set = signal_processor_runner.run(data_set)
        self.assertIs(data_set.data_arrays['x'], new_data_set.data_arrays['x'])
        self.assertTrue(array_equal(new_data_set.data_arrays['x'], array([1, 2, 3, 4, 5])))

    def test_run_process_single_signal_processor(self):
        data_set = DataSet(data_arrays=DataArray('x', 'x', preset_data=array([1, 2, 3, 4, 5])))

        class PlusOneSignalProcessor(SignalProcessorInterface):
            def __init__(self):
                self._signal_data = None

            def run_process(self, signal_data: DataSet) -> DataSet:
                self._signal_data = signal_data
                signal_data.data_arrays['x'] += 1
                return signal_data

        signal_processor_runner = SignalProcessorRunner()
        signal_processor_runner.add_signal_processor(PlusOneSignalProcessor())

        new_data_set = signal_processor_runner.run(data_set)
        self.assertIs(data_set.data_arrays['x'], new_data_set.data_arrays['x'])
        self.assertTrue(array_equal(new_data_set.data_arrays['x'], array([2, 3, 4, 5, 6])))

    def test_run_process_multiple_signal_processors(self):
        data_set = DataSet(data_arrays=DataArray('x', 'x', preset_data=array([1, 2, 3, 4, 5])))

        class PlusOneSignalProcessor(SignalProcessorInterface):
            def __init__(self):
                self._signal_data = None

            def run_process(self, signal_data: DataSet) -> DataSet:
                self._signal_data = signal_data
                signal_data.data_arrays['x'] += 1
                return signal_data

        class TimesTwoSignalProcessor(SignalProcessorInterface):
            def __init__(self):
                self._signal_data = None

            def run_process(self, signal_data: DataSet) -> DataSet:
                self._signal_data = signal_data
                signal_data.data_arrays['x'] *= 2
                return signal_data

        signal_processor_runner = SignalProcessorRunner()
        signal_processor_runner.add_signal_processor(PlusOneSignalProcessor())
        signal_processor_runner.add_signal_processor(TimesTwoSignalProcessor())

        new_data_set = signal_processor_runner.run(data_set)

        self.assertIs(data_set.data_arrays['x'], new_data_set.data_arrays['x'])
        self.assertTrue(array_equal(new_data_set.data_arrays['x'], array([4, 6, 8, 10, 12])))
