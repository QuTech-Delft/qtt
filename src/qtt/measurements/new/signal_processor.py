from typing import List

from qilib.data_set import DataSet

from qtt.measurements.new.interfaces.signal_processor_interface import SignalProcessorInterface


class SignalProcessor:
    """ A class for post-processing measurement data. """

    def __init__(self) -> None:
        self._signal_processors: List[SignalProcessorInterface] = []

    def add_signal_processor(self, signal_processor: SignalProcessorInterface) -> None:
        """ Add a signal processor.

            Args:
                signal_processor: A signal processor
        """

        self._signal_processors.append(signal_processor)

    def run_processes(self, signal_data: DataSet) -> DataSet:
        """ Post-process measurement data using the known signal processors.

            Args:
                signal_data: The measurement data
        """

        for signal_processor in self._signal_processors:
            signal_data = signal_processor.run_process(signal_data)

        return signal_data
