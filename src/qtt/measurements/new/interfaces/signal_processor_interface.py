from abc import ABC, abstractmethod

from qilib.data_set import DataSet


class SignalProcessorInterface(ABC):
    """ An interface for post-processing measurement data. """

    @abstractmethod
    def run_process(self, signal_data: DataSet) -> None:
        """ The post-processing function.

            Args:
                signal_data: The measurement data
        """
