import numpy as np

from qtt.instrument_drivers.virtual_awg.setting import Setting
from qtt.instrument_drivers.virtual_awg.awgs.common import AwgCommon, AwgCommonError


class Tektronix5014C_AWG(AwgCommon):

    def __init__(self, awg):
        super('Tektronix_AWG5014', channels=[1, 2, 3, 4], markers=[1, 2])
        if type(awg).__name__ is not self._awg_name:
            raise AwgCommonError('The AWG does not correspond with {}'.format(self._awg_name))
        self.__settings = {'sampling_rate': Setting('GS/s', 1.0e9, 1.0e7, 1.2e9),
                           'marker_delay': Setting('ns', 0.0, 0.0, 1.0),
                           'marker_low': Setting('V', 0.0, -1.0, 2.6),
                           'marker_high': Setting('V', 1.0, -0.9, 2.7),
                           'amplitudes': Setting('V', 1.0, 0.02, 4.5),
                           'offset': Setting('V', 0, -2.25, 2.25)}
        self.__awg = awg
        self.update_settings()

    @property
    def get(self):
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
        if channels not in self._channel_numbers:
            raise AwgCommonError("Invalid channel numbers {}".format(channels))
        [self.__awg.set('ch{}_state'.format(ch), 1) for ch in channels]

    def disable_outputs(self, channels=None):
        if not channels:
            channels = self._channel_numbers
        if channels not in self._channel_numbers:
            raise AwgCommonError("Invalid channel numbers {}".format(channels))
        [self.__awg.set('ch{}_state'.format(ch), 0) for ch in channels]

    def update_settings(self):
        self.set_sampling_rate(self.__settings['sampling_rate'].value)
        self.set_gain(self.__settings['amplitudes'].value)

    def change_setting(self, key, value):
        if key not in self.__settings.keys():
            raise AwgCommonError('Invalid setting {}'.format(key))
        is_updated = self.__settings[key].update_value(value)
        if not is_updated:
            setting = self.__settings[key]
            raise AwgCommonError('Invalid value {}={} range({}, {})'.format(key,
                                 value, setting.minimum, setting.maximum))
        self.update_settings()

    def set_mode(self, mode):
        modes = ['CONT', 'SEQ']
        if mode not in modes:
            raise AwgCommonError('Invalid AWG mode ({})'.format(mode))
        self.__awg.set('run_mode', 'CONT')

    def get_mode(self):
        self.__awg.get('run_mode')

    def set_sampling_rate(self, sampling_rate):
        self.__awg.set('clock_freq', sampling_rate)

    def get_sampling_rate(self, sampling_rate):
        self.__awg.get('clock_freq')

    def set_gain(self, channel, gain):
        return self.__awg.set('ch{0}_amp'.format(channel), gain)

    def get_gain(self, channel):
        return self.__awg.get('ch{0}_amp'.format(channel))

    def set_sequence(self, channels, sequence):
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

    def delete_sequence(self, channel):
            self.__set_sequence_length(0)

    def upload_waveforms(self, names, waveforms):
        pack_count = len(names)
        packed_waveforms = dict()
        [wfs, m1s, m2s] = list(map(list, zip(*waveforms)))
        for i in range(pack_count):
            name = names[i]
            package = self.__awg.pack_waveform(wfs[i], m1s[i], m2s[i])
            packed_waveforms[name] = package
        offset = self.__settings['offset'].value
        amplitude = self.__settings['amplitudes'].value
        marker_low = self.__settings['marker_low'].value
        marker_high = self.__settings['marker_high'].value
        channel_cfg = {'ANALOG_METHOD_1': 1, 'CHANNEL_STATE_1': 1, 'ANALOG_AMPLITUDE_1': amplitude,
                       'ANALOG_OFFSET_1': offset,
                       'MARKER1_METHOD_1': 2, 'MARKER1_LOW_1': marker_low, 'MARKER1_HIGH_1': marker_high,
                       'MARKER2_METHOD_1': 2, 'MARKER2_LOW_1': marker_low, 'MARKER2_HIGH_1': marker_high,

                       'ANALOG_METHOD_2': 1, 'CHANNEL_STATE_2': 1, 'ANALOG_AMPLITUDE_2': amplitude,
                       'ANALOG_OFFSET_2': offset,
                       'MARKER1_METHOD_2': 2, 'MARKER1_LOW_2': marker_low, 'MARKER1_HIGH_2': marker_high,
                       'MARKER2_METHOD_2': 2, 'MARKER2_LOW_2': marker_low, 'MARKER2_HIGH_2': marker_high,

                       'ANALOG_METHOD_3': 1, 'CHANNEL_STATE_3': 1, 'ANALOG_AMPLITUDE_3': amplitude,
                       'ANALOG_OFFSET_3': offset,
                       'MARKER1_METHOD_3': 2, 'MARKER1_LOW_3': marker_low, 'MARKER1_HIGH_3': marker_high,
                       'MARKER2_METHOD_3': 2, 'MARKER2_LOW_3': marker_low, 'MARKER2_HIGH_3': marker_high,

                       'ANALOG_METHOD_4': 1, 'CHANNEL_STATE_4': 1, 'ANALOG_AMPLITUDE_4': amplitude,
                       'ANALOG_OFFSET_4': offset,
                       'MARKER1_METHOD_4': 2, 'MARKER1_LOW_4': marker_low, 'MARKER1_HIGH_4': marker_high,
                       'MARKER2_METHOD_4': 2, 'MARKER2_LOW_4': marker_low, 'MARKER2_HIGH_4': marker_high}
        file_name = 'costum_awg_file.awg'
        self.__awg.visa_handle.write('MMEMory:CDIRectory "C:\\Users\\OEM\\Documents"')
        awg_file = self.__awg.generate_awg_file(packed_waveforms, np.array([]), [], [], [], [], channel_cfg)
        self.__awg.send_awg_file(file_name, awg_file)
        current_dir = self.__awg.visa_handle.query('MMEMory:CDIRectory?')
        current_dir = current_dir.replace('"', '')
        current_dir = current_dir.replace('\n', '\\')
        self.__awg.load_awg_file('{0}{1}'.format(current_dir, file_name))

    def delete_waveforms(self):
        self.__awg.delete_all_waveforms_from_list()

    def __set_sequence_length(self, count):
        self.__awg.write('SEQuence:LENGth {0}'.format(count))

    def __get_sequence_length(self):
        row_count = self.__awg.ask('SEQuence:LENGth?')
        return int(row_count)
