import numpy as np

from qcodes import Parameter
from qcodes.utils.validators import Numbers
from qtt.instrument_drivers.virtualAwg.awgs.common import (AwgCommon,
                                                           AwgCommonError)

try:
    from qcodes.instrument_drivers.Keysight.M3201A import Keysight_M3201A
except ImportError:
    Keysight_M3201A = None  # type: ignore


class KeysightM3202A_AWG(AwgCommon):

    def __init__(self, awg):
        super().__init__('Keysight_M3201A', channel_numbers=[1, 2, 3, 4], marker_numbers=[1])
        if not Keysight_M3201A:
            raise AwgCommonError('The Keysight SD drivers can not be found!')
        if type(awg).__name__ is not self._awg_name:
            raise AwgCommonError('The AWG does not correspond with {}'.format(self._awg_name))
        self.__settings = [Parameter(name='enabled_outputs', initial_value=0b0000,
                                     set_cmd=None),
                           Parameter(name='amplitude', unit='Volt', initial_value=1.0,
                                     vals=Numbers(0.0, 1.5), set_cmd=None),
                           Parameter(name='offset', unit='seconds', initial_value=0.0,
                                     vals=Numbers(0.0, 2.0), set_cmd=None),
                           Parameter(name='wave_shape', initial_value=6,
                                     set_cmd=None),
                           Parameter(name='delay', unit='seconds', initial_value=0,
                                     vals=Numbers(0, 10), set_cmd=None),
                           Parameter(name='auto_trigger', unit='', initial_value=0,
                                     set_cmd=None),
                           Parameter(name='cycles', initial_value=0,
                                     set_cmd=None),
                           Parameter(name='prescaler', initial_value=10,
                                     set_cmd=None)]
        self.__awg = awg

    @property
    def fetch_awg(self):
        return self.__awg

    def run(self):
        bit_mask = self.retrieve_setting('enabled_outputs')
        self.__awg.awg_start_multiple(bit_mask)

    def stop(self):
        bit_mask = 0b1111
        self.__awg.awg_stop_multiple(bit_mask)

    def reset(self):
        self.__awg.awg.resetAWG()

    def enable_outputs(self, channels=None):
        if not channels:
            channels = self._channel_numbers
        if not all([ch in self._channel_numbers for ch in channels]):
            raise AwgCommonError("Invalid channel numbers {}".format(channels))
        bit_mask = self.retrieve_setting('enabled_outputs')
        for channel in channels:
            bit_mask = bit_mask | 0b1 << (channel - 1)
        self.change_setting('enabled_outputs', bit_mask)

    def disable_outputs(self, channels=None):
        if not channels:
            channels = self._channel_numbers
        if not all([ch in self._channel_numbers for ch in channels]):
            raise AwgCommonError("Invalid channel numbers {}".format(channels))
        bit_mask = self.retrieve_setting('enabled_outputs')
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
        raise ValueError('The sample rate can only be set via the prescaler!')

    def retrieve_sampling_rate(self):
        prescaler = self.retrieve_setting('prescaler')
        if prescaler == 0:
            return 1e9
        elif prescaler == 1:
            return 2e8
        else:
            return 1e8 / prescaler

    def update_gain(self, gain):
        self.change_setting('amplitude', gain)

    def retrieve_gain(self):
        return self.retrieve_setting('amplitude')

    def upload_waveforms(self, sequence_names, sequence_channels, sequence_items, reload=True):
        channel_numbers = [ch[0] for ch in sequence_channels]
        if not all([ch in self._channel_numbers for ch in channel_numbers]):
            raise AwgCommonError("Invalid channel numbers {}".format(channel_numbers))
        wave_number = 0
        delay = self.retrieve_setting('delay')
        cycles = self.retrieve_setting('cycles')
        prescaler = self.retrieve_setting('prescaler')
        auto_trigger = self.retrieve_setting('auto_trigger')
        wave_shape = self.retrieve_setting('wave_shape')
        amplitude = self.retrieve_setting('amplitude')
        offset = self.retrieve_setting('offset')
        for (channel_number, sequence) in zip(channel_numbers, sequence_items):
            wave_object = Keysight_M3201A.new_waveform_from_double(0, np.array(sequence))
            self.__awg.load_waveform(wave_object, wave_number)
            self.__awg.set('wave_shape_channel_%d' % channel_number, wave_shape)
            self.__awg.set('amplitude_channel_%d' % channel_number, amplitude)
            self.__awg.set('offset_channel_%d' % channel_number, offset)
            self.__awg.awg_queue_waveform(channel_number, wave_number, auto_trigger, delay, cycles, prescaler)
            wave_number += 1

    def retrieve_waveforms(self):
        raise NotImplementedError

    def delete_waveforms(self):
        for channel_nr in self._channel_numbers:
            self.__awg.awg_flush(channel_nr)
