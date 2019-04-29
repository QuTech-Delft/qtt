from typing import List

from qilib.data_set import DataSet

from qtt.measurements.post_processing.interfaces.signal_processor_interface import SignalProcessorInterface


class SignalProcessorRunner:
    """ A class for post-processing measurement data. """

    def __init__(self) -> None:
        self._signal_processors: List[SignalProcessorInterface] = []

    def add_signal_processor(self, signal_processor: SignalProcessorInterface) -> None:
        """ Add a signal processor.

            Args:
                signal_processor: A signal processor
        """

        if not isinstance(signal_processor, SignalProcessorInterface):
            raise TypeError('signal_processor must be of type SignalProcesorInterface')

        self._signal_processors.append(signal_processor)

    def run(self, signal_data: DataSet) -> DataSet:
        """ Post-process measurement data using the known signal processors.

            Args:
                signal_data: The measurement data
        """

        for signal_processor in self._signal_processors:
            signal_data = signal_processor.run_process(signal_data)

        return signal_data
