""" Interface for devices to acquire data."""

from abc import ABC, abstractmethod
from typing import List

from qilib.data_set import DataArray
from qilib.utils import PythonJsonStructure


class AcquisitionInterface(ABC):
    """ An interface which contains the functionality for collecting data using a acquisition device."""

    def __init__(self, address: str) -> None:
        """ Creates and connects the acquisition device from the given address.

        Args:
            address: The unique device identifier.
        """
        self._address = address

    @abstractmethod
    def initialize(self, configuration: PythonJsonStructure) -> None:
        """ Initializes the readout device by applying the configuration.

        Args:
            configuration: A structure with all default settings needed
                           for acquiring raw-data from the readout device.
        """

    @abstractmethod
    def prepare_acquisition(self) -> None:
        """ Updates the settings required for acquisition.

            This method should be called after initializing the acquisition
            device and before reading out the device with acquire.
        """

    @abstractmethod
    def acquire(self, number_of_records: int, timeout: float) -> List[DataArray]:
        """ Reads raw-data from the acquisition device.

            This method should be called after initialising and preparing the acquisition.

        Args:
            number_of_records: The number of records collected.
            timeout: The maximum period in seconds to acquire records.

        Returns:
            A list with the collected scope records.
        """

    @abstractmethod
    def finalize_acquisition(self) -> None:
        """ Restores the settings after acquiring records with the device.

            This function should be called after acquiring with the readout device.
        """
