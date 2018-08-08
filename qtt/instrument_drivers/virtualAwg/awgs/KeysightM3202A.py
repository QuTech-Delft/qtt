import numpy as np

from qcodes import Parameter
from qcodes.utils.validators import Numbers, OnOff
from qtt.instrument_drivers.virtualAwg.awgs.common import AwgCommon, AwgCommonError


class KeysightM3202A_AWG(AwgCommon):

    def __init__(self, awg):
        super('M3202A', channels=[1, 2, 3, 4], markers=[1])
        if type(awg).__name__ is not self._awg_name:
            raise AwgCommonError('The AWG does not correspond with {}'.format(self._awg_name))
        self.__settings = [Parameter(name='enabled_outputs', initial_value=0b0000,
                                     set_cmd=None),
                           Parameter(name='sampling_rate', unit='GS/s', initial_value=1.0e9,
                                     vals=Numbers(1.0e7, 1.2e9), set_cmd=None),
                           Parameter(name='amplitude', unit='Volt', initial_value=1.0,
                                     vals=Numbers(0.0, 2.0), set_cmd=None),
                           Parameter(name='offset', unit='seconds', initial_value=0.0,
                                     vals=Numbers(0.0, 2.0), set_cmd=None),
                           Parameter(name='arb_wave', initial_value=6,
                                     set_cmd=None),
                           Parameter(name='delay', unit='seconds', initial_value=0.0,
                                     vals=Numbers(0.0, 10.0), set_cmd=None),
                           Parameter(name='auto_trigger', unit='', initial_value=0,
                                     set_cmd=None),
                           Parameter(name='cycles', initial_value=10000,
                                     set_cmd=None),
                           Parameter(name='prescaler', initial_value=5,
                                     set_cmd=None)]
        self.__awg = awg

    @property
    def fetch_awg(self):
        return self.__awg

    def run(self):
        bit_mask = self.retrieve_setting('enabled_outputs').get()
        self.__awg.awg_start_multiple(bit_mask)

    def stop(self):
        bit_mask = 0b0000
        self.__awg.awg_stop_multiple(bit_mask)

    def reset(self):
        self.__awg.awg.resetAWG()

    def enable_outputs(self, channels=None):
        if not channels:
            channels = self._channel_numbers
        if not all([ch in self._channel_numbers for ch in channels]):
            raise AwgCommonError("Invalid channel numbers {}".format(channels))
        bit_mask = self.retrieve_setting('enabled_outputs').get()
        for channel in channels:
            bit_mask = bit_mask | 0b1 << (channel - 1)
        self.change_setting('enabled_outputs', bit_mask)

    def disable_outputs(self, channels=None):
        if not channels:
            channels = self._channel_numbers
        if not all([ch in self._channel_numbers for ch in channels]):
            raise AwgCommonError("Invalid channel numbers {}".format(channels))
        bit_mask = self.retrieve_setting('enabled_outputs').get()
        for channel in channels:
            bit_mask = bit_mask ^ 0b1 << (channel - 1)
        self.change_setting('enabled_outputs', bit_mask)

    def change_setting(self, name, value):
        index = next(i for i, p in enumerate(self.__settings) if p.name == name)
        self.__settings[index].set(value)

    def retrieve_setting(self, name):
        index = next(i for i, p in enumerate(self.__settings) if p.name == name)
        return self.__settings[index].get()

    def update_running_mode(self, mode):
        raise NotImplementedError

    def retrieve_running_mode(self):
        raise NotImplementedError

    def update_sampling_rate(self, sampling_rate):
        raise NotImplementedError

    def retrieve_sampling_rate(self):
        raise NotImplementedError

    def update_gain(self, gain):
        raise NotImplementedError

    def retrieve_gain(self):
        raise NotImplementedError

    def upload_waveforms(self, sequence_channels, sequence_names, sequence_items, reload=True):
        if not all([ch in self._channel_numbers for ch in sequence_channels]):
            raise AwgCommonError("Invalid channel numbers {}".format(sequence_channels))

    def retrieve_waveforms(self):
        raise NotImplementedError

    def delete_waveforms(self):
        for channel_nr in self._channel_numbers:
            self.__awg.awg_flush(channel_nr)
