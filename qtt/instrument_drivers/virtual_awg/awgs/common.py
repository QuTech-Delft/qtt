from abc import ABC, abstractmethod


class AwgCommonError(Exception):
    """Exception for a specific error related to the AWG common functionality."""


class AwgCommon(ABC):

    def __init__(self, name, channel_numbers, marker_numbers):
        self._awg_name = name
        self._channel_numbers = channel_numbers
        self._channel_count = len(channel_numbers)
        self._marker_numbers = marker_numbers
        self._marker_count = len(marker_numbers)

    @property
    @abstractmethod
    def get(self):
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
    def change_setting(self, setting, value):
        raise NotImplementedError

    @abstractmethod
    def update_settings(self):
        raise NotImplementedError

    @abstractmethod
    def set_mode(self, mode):
        raise NotImplementedError

    @abstractmethod
    def get_mode(self):
        raise NotImplementedError

    @abstractmethod
    def set_sampling_rate(self, sampling_rate):
        raise NotImplementedError

    @abstractmethod
    def get_sampling_rate(self, sampling_rate):
        raise NotImplementedError

    @abstractmethod
    def set_gain(self, gain):
        raise NotImplementedError

    @abstractmethod
    def get_gain(self):
        raise NotImplementedError

    @abstractmethod
    def set_sequence(self, channel, sequence):
        raise NotImplementedError

    @abstractmethod
    def get_sequence(self, channel):
        raise NotImplementedError

    @abstractmethod
    def upload_waveforms(self, names, waveforms):
        raise NotImplementedError

    @abstractmethod
    def delete_waveforms(self):
        raise NotImplementedError
