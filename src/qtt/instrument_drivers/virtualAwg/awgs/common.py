from abc import ABC, abstractmethod


class AwgCommonError(Exception):
    """Exception for a specific error related to the AWG common functionality."""


class AwgCommon(ABC):

    def __init__(self, name, channel_numbers, marker_numbers):
        """ Contains the common functionality for each AWG to be controlled by the virtual AWG.

        Args:
            name (str): The name of the AWG class e.g. Tektronix_AWG5014.
            channel_numbers (List[int]): The channel numbers or identifiers of the AWG.
            marker_numbers (List[int]): The markers numbers or identifiers of the AWG.
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
        """ Enables the main output of the AWG."""
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        """ Disables the main output of the AWG."""
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """ Resets the AWG to its initialization state."""
        raise NotImplementedError

    @abstractmethod
    def enable_outputs(self, channels=None):
        """ Enables the given channel(s) of the AWG. A run command is required to turn on
            the enabled channel(s). No channels argument sets all the channels to enabled.

        Args:
            channels (int, list): The output channel number(s) or identifier(s).
                                  All channels will be enabled if no argument is given.
        """
        raise NotImplementedError

    @abstractmethod
    def disable_outputs(self, channels=None):
        """ Disables the given channel(s) of the AWG. A run command is required to turn off
            the enabled channel(s). No channels argument sets all the channels to disabled.

        Args:
            channels (int, list): The output channel number(s) or identifier(s).
                                  All channels will be disabled if no argument is given.
        """
        raise NotImplementedError

    @abstractmethod
    def change_setting(self, name, value):
        """ Updates a setting of the underlying AWG. The default settings are set during the
            constructing of the AWG.

        Args:
            name (str): The name of the setting, e.g. 'amplitude'
            value (Any): the value to set the setting, e.g. 2.0 V.
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve_setting(self, name):
        """ Gets a setting from the AWG.

        Args:
            setting: The name of the setting, e.g. 'amplitude'

        Returns:
            Any: The value of the setting, e.g. 2.0 V.
        """
        raise NotImplementedError

    @abstractmethod
    def update_running_mode(self, mode):
        """ An AWG has certain running modes for the output channels. This function
            sets the running mode.

        Args:
            mode (Any): The running mode of the AWG, e.g. continues, sequencing.
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve_running_mode(self):
        """ An AWG has certain running modes for the output channels. This functions
            gets the running mode.

        Returns:
            String: The running mode of the AWG, e.g. continues, sequencing.
        """
        raise NotImplementedError

    @abstractmethod
    def update_sampling_rate(self, sampling_rate):
        """ Sets the sampling rate of the AWG.

        Args:
            sampling_rate (int): The number of samples the AWG outputs per second.
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve_sampling_rate(self):
        """ Gets the number of samples the AWG outputs per second.

        Returns:
            Int: The number of samples the AWG outputs per second.
        """
        raise NotImplementedError

    @abstractmethod
    def update_gain(self, gain):
        """ Sets the gain of all AWG output channels.

        Args:
            gain (float): The amplitude of the output channels in arbritrary units.
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve_gain(self):
        """ Gets the gain of all AWG output channels in arbritrary units.

        Returns:
            Float: The amplitude of the output channels in arbritrary units.
        """
        raise NotImplementedError

    @abstractmethod
    def upload_waveforms(self, sequence_names, sequence_channels, sequence_items, reload=True):
        """ Sends the sequences to the AWG.

            Args:
                sequence_names (str, list): A list with the name of the sequence for each sequence.
                sequence_channels (int, list): A list with the channel for each sequence.
                sequence_items (Sequence, list): The Sequencer sequences.
                reload (bool): Reload all the sequences if True else only change the sequence order.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_waveforms(self):
        """ Deletes and removes all the upload waveforms."""
        raise NotImplementedError
