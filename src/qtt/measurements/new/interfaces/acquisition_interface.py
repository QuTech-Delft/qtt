from abc import ABC, abstractmethod
from typing import Any, List, Optional

from qilib.data_set import DataSet
from qilib.utils import PythonJsonStructure


class AcquisitionInterface(ABC):
    """ An interface which contains the functionality for collecting data using a acquisition device."""

    @abstractmethod
    def __init__(self, address: str) -> None:
        """ Creates and connects the acquisition device from the given address.

        Args:
            address: The unique device identifier.
        """

    @abstractmethod
    def initialize(self, configuration: PythonJsonStructure) -> None:
        """ Initializes the readout device by applying the configuration.

        Args:
            configuration: A structure with all default settings needed
                           for acquiring raw-data from the readout device.
        """

    @abstractmethod
    def prepare_acquisition(self) -> None:
        """ Updates the settings required for acquision.

            This function should be called after initializing the acquisition
            device and before reading out the device with acquire.
        """

    @abstractmethod
    def acquire(self, data_set: DataSet, number_of_records: Optional[int], timeout: Optional[int]):
        """ Reads raw-data from the acquisition device.

        Args:
            data_set: A data set with setpoint x-axis and collected acquired raw-data.
            number_of_records: The number of traces collected.
            timeout: The maximum period in seconds to acquire traces.
        """

    @abstractmethod
    def finalize_acquisition(self) -> None:
        """ Restores the settings after acquiring traces with the device.

            This function should be called after initializing the acquisition
            device and before reading out the device with acquire.
        """
