from abc import ABC, abstractmethod
from typing import Any, List

from qtt.measurements.new.interfaces import AcquisitionInterface

class AcquisitionScopeInterface(AcquisitionInterface):
    """ An interface which contains the functionality for a acquisition device as a oscilloscope."""

    @property
    @abstractmethod
    def number_of_averages(self) -> int:
        """ The number of averages to take during a acquisition."""

    @property
    @abstractmethod
    def input_range(self) -> List[float]:
        """ The input range of the channels."""

    @property
    @abstractmethod
    def sample_rate(self) -> float:
        """ The sample rate of the acquisition device."""

    @property
    @abstractmethod
    def period(self) -> float:
        """ Sets the measuring period of the acquisition."""

    @property
    @abstractmethod
    def trigger_enabled(self) -> bool:
        """ Turns the external triggering on or off."""

    @abstractmethod
    def set_trigger_settings(self, channel: int, level: float, slope: str, delay: float) -> None:
        """ Updates the input trigger settings.

        Args:
            channel: The channel to trigger the acquision on.
            level: The trigger-level of the trigger.
            slope: The slope of the trigger.
            delay: The delay between getting a trigger and acquiring.
        """

    @property
    @abstractmethod
    def enabled_channels(self) -> List[int]:
        """ Enables the provides channels and turns off all others."""

    @abstractmethod
    def set_input_signal(self, channel: int, attribute: str) -> None:
        """ Adds an input channel to the scope.

        Args:
            channel: The input channel number.
            attrbutes: The input signal to acquire.
        """
