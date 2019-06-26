import numpy as np

from qcodes import Parameter
from qcodes.utils.validators import Numbers
from qtt.instrument_drivers.virtualAwg.awgs.common import AwgCommon, AwgCommonError


class Tektronix5014C_AWG(AwgCommon):

    def __init__(self, awg):
        super().__init__('Tektronix_AWG5014', channel_numbers=[1, 2, 3, 4], marker_numbers=[1, 2])
        if type(awg).__name__ is not self._awg_name:
            raise AwgCommonError('The AWG does not correspond with {}'.format(self._awg_name))
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
    def fetch_awg(self):
        return self.__awg

    def run(self):
        self.__awg.run()

    def stop(self):
        self.__awg.stop()

    def reset(self):
        self.__awg.reset()

    def enable_outputs(self, channels=None):
        if not channels:
            channels = self._channel_numbers
        if not all([ch in self._channel_numbers for ch in channels]):
            raise AwgCommonError("Invalid channel numbers {}".format(channels))
        [self.__awg.set('ch{}_state'.format(ch), 1) for ch in channels]

    def disable_outputs(self, channels=None):
        if not channels:
            channels = self._channel_numbers
        if not all([ch in self._channel_numbers for ch in channels]):
            raise AwgCommonError("Invalid channel numbers {}".format(channels))
        [self.__awg.set('ch{}_state'.format(ch), 0) for ch in channels]

    def change_setting(self, name, value):
        index = next(i for i, p in enumerate(self.__settings) if p.name == name)
        self.__settings[index].set(value)

    def retrieve_setting(self, name):
        index = next(i for i, p in enumerate(self.__settings) if p.name == name)
        return self.__settings[index].get()

    def update_running_mode(self, mode):
        modes = ['CONT', 'SEQ']
        if mode not in modes:
            raise AwgCommonError('Invalid AWG mode ({})'.format(mode))
        self.__awg.set('run_mode', mode)

    def retrieve_running_mode(self):
        return self.__awg.get('run_mode')

    def update_sampling_rate(self, sampling_rate):
        self.__awg.set('clock_freq', sampling_rate)

    def retrieve_sampling_rate(self):
        return self.__awg.get('clock_freq')

    def update_gain(self, gain):
        [self.__awg.set('ch{0}_amp'.format(ch), gain) for ch in self._channel_numbers]

    def retrieve_gain(self):
        gains = [self.__awg.get('ch{0}_amp'.format(ch)) for ch in self._channel_numbers]
        if not all(g == gains[0] for g in gains):
            raise ValueError('Not all channel amplitudes are equal. Please reset!')
        return gains[0]

    def upload_waveforms(self, sequence_names, sequence_channels, sequence_items, reload=True):
        sequences = (sequence_names, sequence_channels, sequence_items)
        channel_data, waveform_data = Tektronix5014C_AWG.create_waveform_data(*sequences)
        if reload:
            self._upload_waveforms(list(waveform_data.keys()), list(waveform_data.values()))
        self._channel_data = channel_data
        self._set_sequence(list(channel_data.keys()), list(channel_data.values()))

    @staticmethod
    def create_waveform_data(names, channels, items):
        channel_data = dict()
        waveform_data = dict()

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

        return channel_data,waveform_data

    def retrieve_waveforms(self):
        return self._waveform_data if self._waveform_data else None

    def _set_sequence(self, channels, sequence):
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

    def _upload_waveforms(self, names, waveforms, file_name='default.awg'):
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
        self.__awg.load_awg_file('{0}{1}'.format(current_dir, file_name))

    def delete_waveforms(self):
        self.__awg.delete_all_waveforms_from_list()
        self._waveform_data = None

    def __set_sequence_length(self, count):
        self.__awg.write('SEQuence:LENGth {0}'.format(count))

    def __delete_sequence(self):
        self.__set_sequence_length(0)

    def __get_sequence_length(self):
        row_count = self.__awg.ask('SEQuence:LENGth?')
        return int(row_count)
