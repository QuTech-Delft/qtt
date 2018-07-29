from abc import ABC, abstractmethod


class AwgCommonError(Exception):
    """Exception for a specific error related to the AWG common functionality."""


class AwgCommon(ABC):

    def __init__(self, name, channel_numbers, marker_numbers):
        """ Contains the common functionality for each AWG to be controlled by the virtual AWG.

        Args:
            name (str): The name of the AWG class e.g. Tektronix_AWG5014.
            channel_numbers (str): The channel numbers of the AWG.
            marker_numbers (str): The markers numbers of the AWG.
        """
        self._awg_name = name
        self._channel_numbers = channel_numbers
        self._channel_count = len(channel_numbers)
        self._marker_numbers = marker_numbers
        self._marker_count = len(marker_numbers)

    @property
    @abstractmethod
    def fetch_awg(self):
        """ Gets the underlying AWG instance, e.g. the Tektronix_AWG5014 object.

        Returns:
            The AWG instance.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def enable_outputs(self, channels=None):
        raise NotImplementedError

    @abstractmethod
    def disable_outputs(self, channels=None):
        raise NotImplementedError

    @abstractmethod
    def change_setting(self, name, value):
        raise NotImplementedError

    @abstractmethod
    def retrieve_setting(self, name):
        raise NotImplementedError

    @abstractmethod
    def update_running_mode(self, mode):
        raise NotImplementedError

    @abstractmethod
    def retrieve_running_mode(self):
        raise NotImplementedError

    @abstractmethod
    def update_sampling_rate(self, sampling_rate):
        raise NotImplementedError

    @abstractmethod
    def retrieve_sampling_rate(self, sampling_rate):
        raise NotImplementedError

    @abstractmethod
    def update_gain(self, gain):
        raise NotImplementedError

    @abstractmethod
    def retrieve_gain(self):
        raise NotImplementedError

    @abstractmethod
    def upload_waveforms(self, names, waveforms):
        raise NotImplementedError

    @abstractmethod
    def retrieve_waveforms(self):
        raise NotImplementedError

    @abstractmethod
    def delete_waveforms(self):
        raise NotImplementedError
