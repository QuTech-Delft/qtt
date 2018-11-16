from qcodes import Parameter
from qtt.instrument_drivers.virtualAwg.awgs.common import AwgCommon, AwgCommonError


class ZurichInstrumentsHDAWG8(AwgCommon):
    __sampling_rate_map = {0: 2.4e9, 1: 1.2e9, 2: 600e6, 3: 300e6, 4: 150e6, 5: 72e6, 6: 37.50e6, 7: 18.75e6,
                           8: 9.4e6, 9: 4.5e6, 10: 2.34e6, 11: 1.2e3, 12: 586e3, 13: 293e3}

    def __init__(self, awg, awg_number=0):
        """ Implements the common functionality of the AwgCommon for the Zurich Instruments HDAWG8 to be controlled by
        the virtual AWG.

        Note:
            Channels are zero based so channel '0' is output '1' on the physical device.

        Note:
            This backend is setup to work with grouping 1x8 where one awg controls 8 outputs.

        Args:
            awg (ZIHDAWG8): Instance of the QCoDeS ZIHDAWG8 driver.
            awg_number (int): The number of the AWG that is to be controlled. The ZI HDAWG8 has 4 AWGs and the default
                              one is the first one (index 0).
        """
        super().__init__('ZIHDAWG8', channel_numbers=list(range(0, 8)),
                         marker_numbers=list(range(0, 8)))
        if type(awg).__name__ is not self._awg_name:
            raise AwgCommonError('The AWG does not correspond with {}'.format(self._awg_name))
        self.__awg = awg
        self.__awg_number = awg_number
        self.__settings = {'sampling_rate': Parameter(name='sampling_rate', unit='GS/s', initial_value=2.4e9,
                                                      set_cmd=self.update_sampling_rate,
                                                      get_cmd=self.retrieve_sampling_rate)}

    @property
    def fetch_awg(self):
        return self.__awg

    def run(self):
        self.__awg.start_awg(self.__awg_number)

    def stop(self):
        self.__awg.stop_awg(self.__awg_number)

    def reset(self):
        raise NotImplementedError

    def enable_outputs(self, channels=None):
        if channels is None:
            channels = self._channel_numbers
        if not all([ch in self._channel_numbers for ch in channels]):
            raise AwgCommonError("Invalid channel numbers {}".format(channels))
        [self.__awg.enable_channel(ch) for ch in channels]

    def disable_outputs(self, channels=None):
        if channels is None:
            channels = self._channel_numbers
        if not all([ch in self._channel_numbers for ch in channels]):
            raise AwgCommonError("Invalid channel numbers {}".format(channels))
        [self.__awg.disable_channel(ch) for ch in channels]

    def change_setting(self, name, value):
        if name not in self.__settings:
            raise ValueError('No such setting: {}'.format(name))
        self.__settings[name].set(value)

    def retrieve_setting(self, name):
        if name not in self.__settings:
            raise ValueError('No such setting: {}'.format(name))
        return self.__settings[name].get()

    def update_running_mode(self, mode):
        raise NotImplementedError

    def retrieve_running_mode(self):
        raise NotImplementedError

    def update_sampling_rate(self, sampling_rate):
        for sampling_rate_key, sampling_rate_value in ZurichInstrumentsHDAWG8.__sampling_rate_map.items():
            if sampling_rate == sampling_rate_value:
                self.__awg.set('awgs_{}_time'.format(self.__awg_number), sampling_rate_key)
                return
        raise ValueError('Sampling rate {} not in available a list of available values: {}'.format(
            sampling_rate, ZurichInstrumentsHDAWG8.__sampling_rate_map))

    def retrieve_sampling_rate(self):
        sampling_rate_key = self.__awg.get('awgs_{}_time'.format(self.__awg_number))
        return ZurichInstrumentsHDAWG8.__sampling_rate_map[sampling_rate_key]

    def update_gain(self, gain):
        [self.__awg.set('sigouts_{}_range'.format(ch), gain) for ch in self._channel_numbers]

    def retrieve_gain(self):
        gains = [self.__awg.get('sigouts_{}_range'.format(ch)) for ch in self._channel_numbers]
        if not all(g == gains[0] for g in gains):
            raise ValueError('Not all channel gains are equal. Please reset!')
        return gains[0]

    def upload_waveforms(self, sequence_names, sequence_channels, sequence_items, reload=True):
        channel_map = {}
        for name, channel, sequence in zip(sequence_names, sequence_channels, sequence_items):
            if len(channel) == 2:
                sequence = sequence.astype(int)
            channel = channel[0] + 1
            self.__awg.waveform_to_csv(name, sequence)
            if channel in channel_map:
                channel_map[channel].append(name)
            else:
                channel_map[channel] = [name]
        wave_infos = []
        for channel, waves in channel_map.items():
            if len(waves) < 2:
                waves.append(None)
            wave_infos.append((channel, *waves))
        sequence_program = self.__awg.generate_csv_sequence_program(wave_infos)
        self.__awg.upload_sequence_program(self.__awg_number, sequence_program)

    def retrieve_waveforms(self):
        raise NotImplementedError

    def delete_waveforms(self):
        pass
