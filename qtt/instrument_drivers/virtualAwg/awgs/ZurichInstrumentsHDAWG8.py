from qcodes import Parameter
from qcodes.utils.validators import Numbers
from qtt.instrument_drivers.virtualAwg.awgs.common import (AwgCommon,
                                                           AwgCommonError)


class ZurichInstruments_HDAWG8(AwgCommon):

    def __init__(self, awg):
        super().__init__('ZI_HDAWG8', channel_numbers=list(range(1, 9)),
                         marker_numbers=list(range(1, 9)))
        if type(awg).__name__ is not self._awg_name:
            raise AwgCommonError('The AWG does not correspond with {}'.format(self._awg_name))
        self.__device_name = awg._dev.device
        self.__daq = awg._dev.daq
        self.__awg = awg

    @property
    def fetch_awg(self):
        return self.__awg

    def run(self):
        command = "{}/awgs/*/enable".format(self.__device_name)
        self.__daq.setInt(command, 1)

    def stop(self):
        command = "{}/awgs/*/enable".format(self.__device_name)
        self.__daq.setInt(command, 0)

    def reset(self):
        raise NotImplementedError

    def enable_outputs(self, channels=None):
        if not channels:
            channels = self._channel_numbers
        if not all([ch in self._channel_numbers for ch in channels]):
            raise AwgCommonError("Invalid channel numbers {}".format(channels))
        name = self.__device_name
        [self.__daq.setInt("/{0}/sigouts/{1}/on".format(name, ch-1), 1) for ch in channels]

    def disable_outputs(self, channels=None):
        if not channels:
            channels = self._channel_numbers
        if not all([ch in self._channel_numbers for ch in channels]):
            raise AwgCommonError("Invalid channel numbers {}".format(channels))
        name = self.__device_name
        [self.__daq.setInt("/{0}/sigouts/{1}/on".format(name, ch-1), 0) for ch in channels]

    def change_setting(self, name, value):
        raise NotImplementedError

    def retrieve_setting(self, name):
        raise NotImplementedError

    def update_running_mode(self, mode):
        raise NotImplementedError

    def retrieve_running_mode(self):
        raise NotImplementedError

    def update_sampling_rate(self, sampling_rate):
        raise NotImplementedError

    def retrieve_sampling_rate(self, sampling_rate):
        raise NotImplementedError

    def update_gain(self, gain):
        raise NotImplementedError

    def retrieve_gain(self):
        raise NotImplementedError

    def upload_waveforms(self, names, waveforms):
        raise NotImplementedError

    def retrieve_waveforms(self):
        raise NotImplementedError

    def delete_waveforms(self):
        raise NotImplementedError

