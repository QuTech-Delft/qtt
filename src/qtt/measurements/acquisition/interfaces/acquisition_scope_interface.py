""" Interface for oscilloscopes or equivalent devices to acquire data."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional

from qtt.measurements.acquisition.interfaces import AcquisitionInterface


class AcquisitionScopeInterface(AcquisitionInterface, ABC):
    """ An interface which contains the functionality for a acquisition device as a oscilloscope."""

    @property
    @abstractmethod
    def number_of_averages(self) -> int:
        """ The number of averages to take during a acquisition."""

    @property
    @abstractmethod
    def input_range(self) -> Tuple[float, float]:
        """ The input range of the channels."""

    @property
    @abstractmethod
    def sample_rate(self) -> float:
        """ The sample rate of the acquisition device."""

    @property
    @abstractmethod
    def period(self) -> float:
        """ The measuring period of the acquisition."""

    @property
    @abstractmethod
    def trigger_enabled(self) -> bool:
        """ The setter sets the external triggering on or off. The getter returns the current trigger value."""

    @property
    @abstractmethod
    def trigger_channel(self) -> str:
        """ The input signal to trigger the acquisition on."""

    @property
    @abstractmethod
    def trigger_level(self) -> float:
        """ The trigger-level of the trigger in Volts."""

    @property
    @abstractmethod
    def trigger_slope(self) -> str:
        """ The edge of the trigger signal to trigger on."""

    @property
    @abstractmethod
    def trigger_delay(self) -> float:
        """ The delay between getting a trigger and acquiring in seconds."""

    @property
    @abstractmethod
    def enabled_channels(self) -> Tuple[int, ...]:
        """ Reports the enabled input channels."""

    @abstractmethod
    def set_input_signal(self, channel: int, attribute: Optional[str]) -> None:
        """ Adds an input channel to the scope.

        Args:
            channel: The input channel number.
            attribute: The input signal to acquire.
        """
