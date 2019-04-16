from typing import List

from qilib.data_set import DataSet

from qtt.measurements.new.interfaces.signal_processor_interface import SignalProcessorInterface


class SignalProcessor:
    """ A class for post-processing measurement data. """

    def __init__(self) -> None:
        self.signal_processors: List[SignalProcessorInterface] = []

    def run_processes(self, signal_data: DataSet) -> None:
        """ Post-process measurement data using the known signal processors.

            Args:
                signal_data: The measurement data
        """

        for signal_processor in self.signal_processors:
            signal_processor.run_process(signal_data)
