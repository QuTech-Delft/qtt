import numpy as np

from qtt.instrument_drivers.virtual_awg.setting import Setting
from qtt.instrument_drivers.virtual_awg.awgs.common import AwgCommon, AwgCommonError


class SimulatedAWG(AwgCommon):

    def __init__(self, awg):
        super().__init__('Simulated_AWG', channel_numbers=[1, 2, 3, 4], marker_numbers=[1, 2])
        self.__gain = [0]*self._channel_count
        self.__sequence = [0]*self._channel_count
        self.__settings = {'sampling_rate': Setting('GS/s', 1.0e9, 1.0e7, 1.2e9),
                           'marker_delay': Setting('ns', 0.0, 0.0, 1.0),
                           'marker_low': Setting('V', 0.0, -1.0, 2.6),
                           'marker_high': Setting('V', 1.0, -0.9, 2.7),
                           'amplitudes': Setting('V', 1.0, 0.02, 4.5),
                           'offset': Setting('V', 0, -2.25, 2.25)}
        self.update_settings()

    @property
    def get(self):
        raise NotImplementedError

    def run(self):
        return

    def stop(self):
        return

    def reset(self):
        return

    def enable_outputs(self, channels=None):
        return

    def disable_outputs(self, channels=None):
        return

    def update_settings(self):
        self.set_sampling_rate(self.__settings['sampling_rate'].value)
        for channel in self._channel_numbers:
            self.set_gain(channel, self.__settings['amplitudes'].value)

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
        assert(type(mode) == str)
        self.__mode = mode

    def get_mode(self):
        return self.__mode

    def set_sampling_rate(self, sampling_rate):
        self.__sampling_rate = sampling_rate

    def get_sampling_rate(self, sampling_rate):
        return self.__sampling_rate

    def set_gain(self, channel, gain):
        self.__gain[channel-1] = gain

    def get_gain(self, channel):
        return self.__gain[channel-1]

    def set_sequence(self, channel, sequence):
        self.__sequence[channel-1] = sequence

    def get_sequence(self, channel):
        return self.__sequence[channel-1]

    def delete_sequence(self, channel):
        self.__sequence_length = 0

    def upload_waveforms(self, names, waveforms):
        return

    def delete_waveforms(self):
        return

    def __set_sequence_length(self, count):
        self._sequence_length = count

    def __get_sequence_length(self):
        return self.__set_sequence_length
