import numpy as np

from qtt.instrument_drivers.virtualawg.awgs.common import AwgCommon, AwgCommonError


class Simulated_AWG(AwgCommon):

    def __init__(self, awg):
        super().__init__('Simulated_AWG', channel_numbers=[1, 2, 3, 4], marker_numbers=[1, 2])
        self.__gain = [0]*self._channel_count
        self.__sequence = [0]*self._channel_count
        self.__settings = None
        self.__awg = awg

    @property
    def fetch_awg(self):
        return self.__awg

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

    def change_setting(self, key, value):
        pass

    def retrieve_setting(self):
        pass

    def update_running_mode(self, mode):
        self.__mode = mode

    def retrieve_running_mode(self):
        return self.__mode

    def update_sampling_rate(self, sampling_rate):
        self.__sampling_rate = sampling_rate

    def retrieve_sampling_rate(self, sampling_rate):
        return self.__sampling_rate

    def update_gain(self, channel, gain):
        self.__gain[channel-1] = gain

    def retrieve_gain(self, channel):
        return self.__gain[channel-1]

    def upload_waveforms(self, sequence_channels, sequence_names, sequences):
        pass

    def retrieve_waveforms(self):
        pass

    def delete_waveforms(self):
        pass
