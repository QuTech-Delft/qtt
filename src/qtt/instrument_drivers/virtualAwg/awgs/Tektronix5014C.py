from typings import Any, Dict, List, Optional, Tuple
import numpy as np

from qcodes import Parameter
from qcodes.utils.validators import Numbers
from qcodes.instrument_drivers.tektronix.AWG5014 import Tektronix_AWG5014
from qtt.instrument_drivers.virtualAwg.awgs.common import AwgCommon, AwgCommonError


class Tektronix5014C_AWG(AwgCommon):

    def __init__(self, awg: Tektronix_AWG5014) -> None:
        super().__init__('Tektronix_AWG5014', channel_numbers=[1, 2, 3, 4], marker_numbers=[1, 2])
        if type(awg).__name__ is not self._awg_name:
            raise AwgCommonError(f'The AWG does not correspond with {self._awg_name}')
        self.__settings = [Parameter(name='sampling_rate', unit='GS/s', initial_value=1.0e9,
                                     vals=Numbers(1.0e7, 1.2e9), set_cmd=None),
                           Parameter(name='marker_delay', unit='ns', initial_value=0.0,
                                     vals=Numbers(0.0, 1.0), set_cmd=None),
                           Parameter(name='marker_low', unit='V', initial_value=0.0,
                                     vals=Numbers(-1.0, 2.6), set_cmd=None),
                           Parameter(name='marker_high', unit='V', initial_value=1.0,
                                     vals=Numbers(-0.9, 2.7), set_cmd=None),
                           Parameter(name='amplitudes', unit='V', initial_value=1.0,
                                     vals=Numbers(0.02, 4.5), set_cmd=None),
                           Parameter(name='offset', unit='V', initial_value=0,
                                     vals=Numbers(-2.25, 2.25), set_cmd=None)]
        self._waveform_data = None
        self.__awg = awg

    @property
    def fetch_awg(self) -> Tektronix_AWG5014:
        return self.__awg

    def run(self) -> None:
        self.__awg.run()

    def stop(self) -> None:
        self.__awg.stop()

    def reset(self) -> None:
        self.__awg.reset()

    def enable_outputs(self, channels: Optional[List[int]] = None) -> None:
        if not channels:
            channels = self._channel_numbers
        if not all([ch in self._channel_numbers for ch in channels]):
            raise AwgCommonError(f'Invalid channel numbers {channels}')
        [self.__awg.set(f'ch{ch}_state', 1) for ch in channels]

    def disable_outputs(self, channels: Optional[List[int]] = None) -> None:
        if not channels:
            channels = self._channel_numbers
        if not all([ch in self._channel_numbers for ch in channels]):
            raise AwgCommonError(f'Invalid channel numbers {channels}')
        [self.__awg.set(f'ch{ch}_state', 0) for ch in channels]

    def change_setting(self, name: str, value: Any) -> None:
        index = next(i for i, p in enumerate(self.__settings) if p.name == name)
        self.__settings[index].set(value)

    def retrieve_setting(self, name: str) -> Any:
        index = next(i for i, p in enumerate(self.__settings) if p.name == name)
        return self.__settings[index].get()

    def update_running_mode(self, mode: str) -> None:
        modes = ['CONT', 'SEQ']
        if mode not in modes:
            raise AwgCommonError(f'Invalid AWG mode ({mode})')
        self.__awg.set('run_mode', mode)

    def retrieve_running_mode(self) -> float:
        return self.__awg.get('run_mode')

    def retrieve_sampling_rate(self) -> float:
        return self.__awg.get('clock_freq')

    def update_gain(self, gain: float) -> None:
        [self.__awg.set(f'ch{ch}_amp', gain) for ch in self._channel_numbers]

    def retrieve_gain(self) -> float:
        gains = [self.__awg.get(f'ch{ch}_amp') for ch in self._channel_numbers]
        if not all(g == gains[0] for g in gains):
            raise ValueError('Not all channel amplitudes are equal. Please reset!')
        return gains[0]

    def upload_waveforms(self, sequence_names: List[str], sequence_channels: List[Tuple[int, ...]],
                         sequence_items: List[np.ndarray], reload: bool = True) -> None:
        sequences = (sequence_names, sequence_channels, sequence_items)
        channel_data, waveform_data = Tektronix5014C_AWG.create_waveform_data(*sequences)
        self._waveform_data = waveform_data
        self._channel_data = channel_data
        if reload:
            self._upload_waveforms(list(waveform_data.keys()), list(waveform_data.values()))

        self._set_sequence(list(channel_data.keys()), list(channel_data.values()))

    @staticmethod
    def create_waveform_data(names: List[str], channels: List[Tuple[int, ...]],
                             items: List[np.ndarray]) -> Tuple[Dict[str, int], Dict[int, List[np.ndarry]]]:
        channel_data: Dict[str, int] = {}
        waveform_data: Dict[int, List[np.ndarry]] = {}

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

    def retrieve_waveforms(self) -> Dict[str, List[np.ndarray]]:
        return self._waveform_data if self._waveform_data else None

    def _set_sequence(self, channels: List[int], sequence: List[str]) -> None:
        if not sequence or len(sequence) != len(channels):
            raise AwgCommonError('Invalid sequence and channel count!')
        if not all(len(idx) == len(sequence[0]) for idx in sequence):
            raise AwgCommonError('Invalid sequence list lengthts!')
        request_rows = len(sequence[0])
        current_rows = self.__get_sequence_length()
        if request_rows != current_rows:
            self.__set_sequence_length(request_rows)
        for row_index in range(request_rows):
            for channel in self._channel_numbers:
                if channel in channels:
                    ch_index = channels.index(channel)
                    wave_name = sequence[ch_index][row_index]
                    self.__awg.set_sqel_waveform(wave_name, channel, row_index + 1)
                else:
                    self.__awg.set_sqel_waveform("", channel, row_index + 1)
        self.__awg.set_sqel_goto_state(request_rows, 1)

    def _upload_waveforms(self, names: List[str], waveforms: List[np.ndarray],
                          file_name: str = 'default.awg') -> None:
        pack_count = len(names)
        packed_waveforms = dict()
        [wfs, m1s, m2s] = list(map(list, zip(*waveforms)))
        for i in range(pack_count):
            name = names[i]
            package = self.__awg.pack_waveform(wfs[i], m1s[i], m2s[i])
            packed_waveforms[name] = package
        offset = self.__settings[5].get()
        amplitude = self.__settings[4].get()
        marker_low = self.__settings[2].get()
        marker_high = self.__settings[3].get()
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
        self.__awg.delete_all_waveforms_from_list()
        self._waveform_data = None

    def __set_sequence_length(self, count: int) -> None:
        self.__awg.write(f'SEQuence:LENGth {count}')

    def __delete_sequence(self) -> None:
        self.__set_sequence_length(0)

    def __get_sequence_length(self) -> int:
        row_count = self.__awg.ask('SEQuence:LENGth?')
        return int(row_count)
