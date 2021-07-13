""" Interface for oscilloscopes or equivalent devices to acquire data."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from qtt.measurements.acquisition.interfaces import AcquisitionInterface


class AcquisitionScopeInterface(AcquisitionInterface, ABC):
    """ An interface which contains the functionality for a acquisition device as a oscilloscope."""

    @property  # type: ignore
    @abstractmethod
    def input_range(self) -> Tuple[float, float]:
        """ The input range of the channels."""

    @input_range.setter  # type: ignore
    @abstractmethod
    def input_range(self, value: Tuple[float, float]) -> None:
        """ Gets the amplitude input range of the channels.

        Args:
            value: The input range amplitude in Volts.
        """

    @property  # type: ignore
    @abstractmethod
    def sample_rate(self) -> float:
        """ The sample rate of the acquisition device."""

    @sample_rate.setter  # type: ignore
    @abstractmethod
    def sample_rate(self, value: float) -> None:
        """ Sets the sample rate of the acquisition device.

        Args:
            value: The sample rate in samples per second.
        """

    @property  # type: ignore
    @abstractmethod
    def period(self) -> float:
        """ The measuring period of the acquisition."""

    @period.setter  # type: ignore
    @abstractmethod
    def period(self, value: float) -> None:
        """ Sets the measuring period of the acquisition.

        Args:
            value: The measuring period in seconds.
        """

    @property  # type: ignore
    @abstractmethod
    def number_of_samples(self) -> int:
        """ The number of samples to take during a acquisition."""

    @number_of_samples.setter  # type: ignore
    @abstractmethod
    def number_of_samples(self, value: int) -> None:
        """ Sets the sample count to take during a acquisition.

        Args:
            value: The number of samples.
        """

    @property  # type: ignore
    @abstractmethod
    def trigger_enabled(self) -> bool:
        """ The setter sets the external triggering on or off. The getter returns the current trigger value."""

    @trigger_enabled.setter  # type: ignore
    @abstractmethod
    def trigger_enabled(self, value: bool) -> None:
        """ Turns the external triggering on or off.

        Args:
            value: The trigger on/off value.
        """

    @property  # type: ignore
    @abstractmethod
    def trigger_channel(self) -> str:
        """ The input signal to trigger the acquisition on."""

    @trigger_channel.setter  # type: ignore
    @abstractmethod
    def trigger_channel(self, channel: str) -> None:
        """ Sets the external triggering channel.

        Args:
            channel: The trigger channel value.
        """

    @property  # type: ignore
    @abstractmethod
    def trigger_level(self) -> float:
        """ The trigger-level of the trigger in Volts."""

    @trigger_level.setter  # type: ignore
    @abstractmethod
    def trigger_level(self, level: float) -> None:
        """ Sets the external triggering level.

        Args:
            level: The external trigger level in Volts.
        """

    @property  # type: ignore
    @abstractmethod
    def trigger_slope(self) -> str:
        """ The edge of the trigger signal to trigger on."""

    @trigger_slope.setter  # type: ignore
    @abstractmethod
    def trigger_slope(self, slope: str) -> None:
        """ Sets the external triggering slope.

        Args:
            slope: The external trigger slope.
        """

    @property  # type: ignore
    @abstractmethod
    def trigger_delay(self) -> float:
        """ The delay between getting a trigger and acquiring in seconds."""

    @trigger_delay.setter  # type: ignore
    @abstractmethod
    def trigger_delay(self, delay: float) -> None:
        """ Sets the delay in seconds between the external trigger and acquisition.

        Args:
            delay: The scope trigger delay in seconds.
        """

    @property  # type: ignore
    @abstractmethod
    def enabled_channels(self) -> Tuple[int, ...]:
        """ Reports the enabled input channels."""

    @enabled_channels.setter  # type: ignore
    @abstractmethod
    def enabled_channels(self, value: Tuple[int, ...]):
        """ Sets the given channels to enabled and turns off all others.

        Args:
            value: The channels which needs to be enabled.
        """

    @abstractmethod
    def set_input_signal(self, channel: int, attribute: Optional[str]) -> None:
        """ Adds an input channel to the scope.

        Args:
            channel: The input channel number.
            attribute: The input signal to acquire.
        """
