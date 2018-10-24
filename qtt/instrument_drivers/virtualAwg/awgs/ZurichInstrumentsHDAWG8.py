from qtt.instrument_drivers.virtualAwg.awgs.common import AwgCommon, AwgCommonError


class ZurichInstrumentsHDAWG8(AwgCommon):
    def __init__(self, awg, awg_number=0):
        super().__init__('ZIHDAWG8', channel_numbers=list(range(1, 9)),
                         marker_numbers=list(range(1, 9)))
        if type(awg).__name__ is not self._awg_name:
            raise AwgCommonError('The AWG does not correspond with {}'.format(self._awg_name))
        self.__awg = awg
        self.__awg_number = awg_number

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
        raise NotImplementedError

    def retrieve_setting(self, name):
        raise NotImplementedError

    def update_running_mode(self, mode):
        raise NotImplementedError

    def retrieve_running_mode(self):
        raise NotImplementedError

    def update_sampling_rate(self, sampling_rate):
        raise NotImplementedError

    def retrieve_sampling_rate(self):
        sampling_rate_map = {0: 2.4e9, 1: 1.2e9, 2: 600e6, 3: 300e6, 4: 150e6, 5: 72e6, 6: 37.50e6, 7: 18.75e6,
                             8: 9.4e6, 9: 4.5e6, 10: 2.34e6, 11: 1.2e3, 12: 586e3, 13: 293e3}
        sample_rate = self.__awg.get('awgs_{}_time'.format(self.__awg_number))
        return sampling_rate_map[sample_rate]

    def update_gain(self, gain):
        raise NotImplementedError

    def retrieve_gain(self):
        gains = [self.__awg.get('sigouts_{}_range'.format(ch)) for ch in self._channel_numbers]
        if not all(g == gains[0] for g in gains):
            raise ValueError('Not all channel amplitudes are equal. Please reset!')
        return gains[0]

    def upload_waveforms(self, sequence_names, sequence_channels, sequence_items, reload=True):
        for name, sequence in zip(sequence_names, sequence_items):
            self.__awg.waveform_to_csv(name, sequence)
        channels = [ch[0] for ch in sequence_channels]
        sequence_program = self.__awg.generate_csv_sequence_program(sequence_names, channels)
        self.__awg.upload_sequence_program(self.__awg_number, sequence_program)

    def retrieve_waveforms(self):
        raise NotImplementedError

    def delete_waveforms(self):
        pass
