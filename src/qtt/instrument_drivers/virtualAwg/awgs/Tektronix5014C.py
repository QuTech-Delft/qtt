import numpy as np
from qcodes import Parameter
from qcodes.instrument_drivers.tektronix.AWG5014 import Tektronix_AWG5014
from qcodes.utils.validators import Numbers

from qtt.instrument_drivers.virtualAwg.awgs.common import AwgCommon, AwgCommonError
from typing import Dict, List, Optional, Tuple


class Tektronix5014C_AWG(AwgCommon):

    def __init__(self, awg: Tektronix_AWG5014) -> None:
        """ The VirtualAWG backend for the Tektronix 5014C AWG.

        Args:
            awg: The Tektronix 5014C AWG instance.

        Raises:
            ValueError: The provided AWG is not of type Tektronix_AWG5014.
        """
        super().__init__('Tektronix_AWG5014', channel_numbers=[1, 2, 3, 4], marker_numbers=[1, 2])
        if type(awg).__name__ is not self._awg_name:
            raise ValueError(f'The AWG type does not correspond with {self._awg_name}')
        self.__settings = [Parameter(name='marker_low', unit='V', initial_value=0.0,
                                     vals=Numbers(-1.0, 2.6), set_cmd=None),
                           Parameter(name='marker_high', unit='V', initial_value=1.0,
                                     vals=Numbers(-0.9, 2.7), set_cmd=None),
                           Parameter(name='amplitudes', unit='V', initial_value=1.0,
                                     vals=Numbers(0.02, 4.5), set_cmd=None),
                           Parameter(name='offset', unit='V', initial_value=0,
                                     vals=Numbers(-2.25, 2.25), set_cmd=None)]
        self.__awg = awg

    @property
    def fetch_awg(self) -> Tektronix_AWG5014:
        """ Return the AWG instance."""
        return self.__awg

    def run(self) -> None:
        """ Enable the AWG outputs.

        This function equals enabling Run button on the AWG.
        """
        self.__awg.run()

    def stop(self) -> None:
        """ Disables the AWG outputs.

        This function equals disabling the Run button on the AWG.
        """
        self.__awg.stop()

    def reset(self) -> None:
        """ Resets the AWG to it's default settings."""
        self.__awg.reset()

    def enable_outputs(self, channels: Optional[List[int]] = None) -> None:
        """ Enables the outputs for the given channels.

        This function equals enabling the CH1, .. CH4 buttons on the AWG.

        Args:
            channels: A list with the channel numbers. All channels are enabled, if no value is given.

        Raises:
            ValueError: If channels contains an invalid channel number.
        """
        if channels is None:
            channels = self._channel_numbers
        if not all([ch in self._channel_numbers for ch in channels]):
            raise ValueError(f'Invalid channel numbers {channels}')
        list(map(lambda channel: self.__awg.set(f'ch{channel}_state', 1), channels))  # type: ignore

    def disable_outputs(self, channels: Optional[List[int]] = None) -> None:
        """ Disables the outputs for the given channels.

        This function equals disabling the CH1, .. CH4 buttons on the AWG.

        Args:
            channels: A list with the channel numbers. All channels are enabled, if no value is given.

        Raises:
            ValueError: If channels contains an invalid channel number.
        """
        if channels is None:
            channels = self._channel_numbers
        if not all([ch in self._channel_numbers for ch in channels]):
            raise ValueError(f'Invalid channel numbers {channels}')
        list(map(lambda channel: self.__awg.set(f'ch{channel}_state', 0), channels))  # type: ignore

    def change_setting(self, name: str, value: float) -> None:
        """ Sets a setting on the AWG. The changeable settings are:
            marker_low, marker_high, amplitudes and offset.

        Args:
            name: The name of the setting.
            value: The value the setting should get.
        """
        index = next(i for i, p in enumerate(self.__settings) if p.name == name)
        self.__settings[index].set(value)

    def retrieve_setting(self, name: str) -> float:
        """ Gets a setting from the AWG. The gettable are:
            marker_low, marker_high, amplitudes and offset.

        Args:
            name: The name of the setting.
        """
        index = next(i for i, p in enumerate(self.__settings) if p.name == name)
        return self.__settings[index].get()

    def update_running_mode(self, mode: str) -> None:
        """ Sets the running mode. The possible modes are the
            continues (CONT) and sequential (SEQ).

        Args:
            mode: Either 'CONT' (continues) or 'SEQ' (sequential).

        Raises:
            ValueError: If the given mode is not 'CONT' (continues) or 'SEQ' (sequential).
        """
        modes = ['CONT', 'SEQ']
        if mode not in modes:
            raise ValueError(f'Invalid AWG mode ({mode})')
        self.__awg.set('run_mode', mode)

    def retrieve_running_mode(self) -> float:
        """ Sets the running mode. The possible modes are the
            continues (CONT) and sequential (SEQ).

        Returns:
            'CONT' or 'SEQ'.
        """
        return self.__awg.get('run_mode')

    def update_sampling_rate(self, sampling_rate: int) -> None:
        """ Sets the sampling rate of the AWG.

        Args:
            sampling_rate: The number of samples the AWG outputs per second.
        """
        return self.__awg.set('clock_freq', sampling_rate)

    def retrieve_sampling_rate(self) -> int:
        """ Gets the sample rate of the AWG.

        Returns:
            The sample rate of the AWG in Samples/second.
        """
        return self.__awg.get('clock_freq')

    def update_gain(self, gain: float) -> None:
        """ Sets the amplitude of the channel outputs.

        The amplitude for all channels are set to the same value using this function.

        Args:
            gain: The amplitude of the output channels.
        """
        list(map(lambda channel: self.__awg.set(f'ch{channel}_amp', gain), self._channel_numbers))  # type: ignore

    def retrieve_gain(self) -> float:
        """ Gets the amplitude for all the output channels.

        Returns:
            The amplitude for all output channels.

        Raises:
            AwgCommonError: If not all channel amplitudes have the same value. Then the settings in
                            the AWG are off and needs to be reset first.
        """
        gains = [self.__awg.get(f'ch{ch}_amp') for ch in self._channel_numbers]
        if not all([g == gains[0] for g in gains]):
            raise AwgCommonError('Not all channel amplitudes are equal. Please reset!')
        return gains[0]

    def upload_waveforms(self, sequence_names: List[str], sequence_channels: List[Tuple[int, ...]],
                         sequence_items: List[np.ndarray], reload: bool = True) -> None:
        """ Uploads the sequence with waveforms to the user defined waveform list.

        Args:
            sequences_names: The names of the waveforms.
            sequence_channels: A list containing the channel numbers to which each waveform belongs.
                               E.g. [(1,), (1, 2)]. The second tuple element corresponds to the marker number.
            sequence_items: A list containing the data for each waveform.
        """
        sequence_input = (sequence_names, sequence_channels, sequence_items)
        channel_data, waveform_data = Tektronix5014C_AWG.create_waveform_data(*sequence_input)
        if reload:
            names = list(waveform_data.keys())
            waveforms = list(waveform_data.values())
            self._upload_waveforms(names, waveforms)

        channels = list(channel_data.keys())
        sequences = list(channel_data.values())
        self._set_sequence(channels, sequences)

    @staticmethod
    def create_waveform_data(names: List[str], channels: List[Tuple[int, ...]],
                             items: List[np.ndarray]) -> Tuple[Dict[int, List[str]], Dict[str, List[np.ndarray]]]:
        """ Transforms the data into the correct waveform data.

            A marker waveform will be merged with the channel waveform if the channels list contain both
            the marker waveform and the output waveform on the same channel.

        Args:
            names: The waveform names.
            channels: A list containing the channel numbers to which each waveform belongs.
                      E.g. [(1,), (1, 2)]. The second tuple element corresponds to the marker number.
            items: A list containing the data for each waveform.

        Returns:
            A tuple with the channel data and the waveform data.
            The channel data contains for each waveform the name and to which channel it belongs.
            The waveform data contains for each waveform the name and the actual waveform.

        Raises:
            ValueError: If the number of elements in names, channels and items do not match.
        """
        channel_data: Dict[int, List[str]] = {}
        waveform_data: Dict[str, List[np.ndarray]] = {}

        if len(names) != len(channels) or len(names) != len(items):
            error_text = 'The waveform input data is not correct!' \
                         f'Length of (names, channels, items) are ({len(names)}, {len(channels)}, {len(items)}.'
            raise ValueError(error_text)

        unique_channels = set([channel for (channel, *_) in channels])
        for unique_channel in unique_channels:
            indices = [channels.index(item) for item in channels if item[0] == unique_channel]
            data_count = len(items[indices[0]])
            data_name = names[indices[0]]

            data = [np.zeros(data_count), np.zeros(data_count), np.zeros(data_count)]
            for index in indices:
                (_, *marker) = channels[index]
                data[marker[0] if marker else 0] = items[index]
                if not marker:
                    data_name = names[index]

            waveform_data[data_name] = data
            channel_data.setdefault(unique_channel, []).append(data_name)

        return channel_data, waveform_data

    def _set_sequence(self, channels: List[int], sequence: List[List[str]]) -> None:
        """ Sets the sequence on the AWG using the user defined waveforms.

        Args:
            channels: A list with channel numbers that should output the waveform on the AWG.
            sequence: A list containing lists with the waveform names for each channel.
                      The outer list determines the number of rows the sequences has.

        Raises:
            ValueError: If the number of channels does not match the element count in the sequence or
                        when the number of waveforms for each channel do not match.
        """
        if len(sequence) != len(channels):
            raise ValueError('Invalid sequence and channel count!')
        if not all(len(idx) == len(sequence[0]) for idx in sequence):
            raise ValueError('Invalid sequence list lengths!')
        request_rows = len(sequence[0])
        current_rows = self.get_sequence_length()
        if request_rows != current_rows:
            self.set_sequence_length(request_rows)
        for row_index in range(request_rows):
            for channel in self._channel_numbers:
                if channel in channels:
                    ch_index = channels.index(channel)
                    wave_name = sequence[ch_index][row_index]
                    self.__awg.set_sqel_waveform(wave_name, channel, row_index + 1)
                else:
                    self.__awg.set_sqel_waveform("", channel, row_index + 1)
        self.__awg.set_sqel_goto_state(request_rows, 1)

    def _upload_waveforms(self, names: List[str], waveforms: List[List[np.ndarray]],
                          file_name: str = 'default.awg') -> None:
        """ Upload the waveforms to the AWG. Creates an AWG file and then uploads it to the AWG.

        Args:
            names: A list with the waveform names.
            waveforms: A list containing the waveform data.
            file_name: The name of the AWG file.
        """
        pack_count = len(names)
        packed_waveforms = dict()
        [wfs, m1s, m2s] = list(map(list, zip(*waveforms)))
        for i in range(pack_count):
            name = names[i]
            package = self.__awg.pack_waveform(wfs[i], m1s[i], m2s[i])
            packed_waveforms[name] = package
        offset = self.retrieve_setting('offset')
        amplitude = self.retrieve_setting('amplitudes')
        marker_low = self.retrieve_setting('marker_low')
        marker_high = self.retrieve_setting('marker_high')
        channel_cfg = {'ANALOG_METHOD_1': 1, 'CHANNEL_STATE_1': 1, 'ANALOG_AMPLITUDE_1': amplitude,
                       'MARKER1_METHOD_1': 2, 'MARKER1_LOW_1': marker_low, 'MARKER1_HIGH_1': marker_high,
                       'MARKER2_METHOD_1': 2, 'MARKER2_LOW_1': marker_low, 'MARKER2_HIGH_1': marker_high,
                       'ANALOG_OFFSET_1': offset,
                       'ANALOG_METHOD_2': 1, 'CHANNEL_STATE_2': 1, 'ANALOG_AMPLITUDE_2': amplitude,
                       'MARKER1_METHOD_2': 2, 'MARKER1_LOW_2': marker_low, 'MARKER1_HIGH_2': marker_high,
                       'MARKER2_METHOD_2': 2, 'MARKER2_LOW_2': marker_low, 'MARKER2_HIGH_2': marker_high,
                       'ANALOG_OFFSET_2': offset,
                       'ANALOG_METHOD_3': 1, 'CHANNEL_STATE_3': 1, 'ANALOG_AMPLITUDE_3': amplitude,
                       'MARKER1_METHOD_3': 2, 'MARKER1_LOW_3': marker_low, 'MARKER1_HIGH_3': marker_high,
                       'MARKER2_METHOD_3': 2, 'MARKER2_LOW_3': marker_low, 'MARKER2_HIGH_3': marker_high,
                       'ANALOG_OFFSET_3': offset,
                       'ANALOG_METHOD_4': 1, 'CHANNEL_STATE_4': 1, 'ANALOG_AMPLITUDE_4': amplitude,
                       'MARKER1_METHOD_4': 2, 'MARKER1_LOW_4': marker_low, 'MARKER1_HIGH_4': marker_high,
                       'MARKER2_METHOD_4': 2, 'MARKER2_LOW_4': marker_low, 'MARKER2_HIGH_4': marker_high,
                       'ANALOG_OFFSET_4': offset}
        self.__awg.visa_handle.write('MMEMory:CDIRectory "C:\\Users\\OEM\\Documents"')
        awg_file = self.__awg.generate_awg_file(packed_waveforms, np.array([]), [], [], [], [], channel_cfg)
        self.__awg.send_awg_file(file_name, awg_file)
        current_dir = self.__awg.visa_handle.query('MMEMory:CDIRectory?')
        current_dir = current_dir.replace('"', '')
        current_dir = current_dir.replace('\n', '\\')
        self.__awg.load_awg_file(f'{current_dir}{file_name}')

    def delete_waveforms(self) -> None:
        """ Clears the user defined waveform list from the AWG."""
        self.__awg.delete_all_waveforms_from_list()

    def delete_sequence(self) -> None:
        """ Clears the sequence from the AWG."""
        self.set_sequence_length(0)

    def set_sequence_length(self, row_count: int) -> None:
        """ Sets the number of rows in the sequence.

        Args:
            row_count: The number of rows in the sequence.
        """
        self.__awg.write(f'SEQuence:LENGth {row_count}')

    def get_sequence_length(self) -> int:
        """ Gets the number of rows in the sequence.

        Returns:
            The number of rows in the sequence.
        """
        row_count = self.__awg.ask('SEQuence:LENGth?')
        return int(row_count)
