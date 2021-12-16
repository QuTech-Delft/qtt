import logging
import os

import numpy as np
import zhinst
from qcodes import Parameter

from qtt.instrument_drivers.virtualAwg.awgs.common import AwgCommon, AwgCommonError

logger = logging.getLogger(__name__)


class ZurichInstrumentsHDAWG8(AwgCommon):
    __sampling_rate_map = {ii: 2.4e9 / 2**ii for ii in range(0, 14)}

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
            raise AwgCommonError(f'The AWG does not correspond with {self._awg_name}')
        self.__awg = awg
        self.__awg_number = awg_number
        self.__settings = {'sampling_rate': Parameter(name='sampling_rate', unit='GS/s',
                                                      set_cmd=self.update_sampling_rate,
                                                      get_cmd=self.retrieve_sampling_rate)}

        self._use_binary_waves = True

    def __str__(self):
        class_name = self.__class__.__name__
        instrument_name = self.__awg.name
        return f'<{class_name} at {hex(id(self))}: {instrument_name}>'

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
            raise AwgCommonError(f"Invalid channel numbers {channels}")
        _ = [self.__awg.enable_channel(ch) for ch in channels]

    def disable_outputs(self, channels=None):
        if channels is None:
            channels = self._channel_numbers
        if not all([ch in self._channel_numbers for ch in channels]):
            raise AwgCommonError(f"Invalid channel numbers {channels}")
        _ = [self.__awg.disable_channel(ch) for ch in channels]

    def change_setting(self, name, value):
        if name not in self.__settings:
            raise ValueError(f'No such setting: {name}')
        self.__settings[name].set(value)

    def retrieve_setting(self, name):
        if name not in self.__settings:
            raise ValueError(f'No such setting: {name}')
        return self.__settings[name].get()

    def update_running_mode(self, mode):
        raise NotImplementedError

    def retrieve_running_mode(self):
        raise NotImplementedError

    def update_sampling_rate(self, sampling_rate):
        for sampling_rate_key, sampling_rate_value in ZurichInstrumentsHDAWG8.__sampling_rate_map.items():
            if sampling_rate == sampling_rate_value:
                self.__awg.set(f'awgs_{self.__awg_number}_time', sampling_rate_key)
                return
        raise ValueError('Sampling rate {} not in available a list of available values: {}'.format(
            sampling_rate, ZurichInstrumentsHDAWG8.__sampling_rate_map))

    def retrieve_sampling_rate(self):
        sampling_rate_key = self.__awg.get(f'awgs_{self.__awg_number}_time')
        return ZurichInstrumentsHDAWG8.__sampling_rate_map[sampling_rate_key]

    def update_gain(self, gain):
        """ Set the gain of the device by setting the range of all channels to two times the gain

        The range is twice the gain under the assumption that the load on the output channels is 50 Ohm. For a high
        impedance load the gain equals the range.
        """
        _ = [self.__awg.set(f'sigouts_{ch}_range', 2 * gain) for ch in self._channel_numbers]

    def retrieve_gain(self):
        gains = [self.__awg.get(f'sigouts_{ch}_range') / 2 for ch in self._channel_numbers]
        if not all(g == gains[0] for g in gains):
            raise ValueError(f'Not all channel gains {gains} are equal. Please reset!')
        return gains[0]

    @staticmethod
    def waveform_to_wave(awg_driver, wave_name: str, waveform: np.ndarray) -> None:
        """
        Write waveforms to a .wave file in the modules data directory so that it
        can be referenced and used in a sequence program. If more than one
        waveform is provided they will be played simultaneously but on separate
        outputs.

        Args:
            wave_name: Name of the wave file, is used by a sequence program.
            waveforms: One or more waveforms that are to be written to a
                CSV file. Note if there are more than one waveforms then they
                have to be of equal length, if not the longer ones will be
                truncated.
        """
        data_dir = awg_driver.awg_module.getString('awgModule/directory')
        wave_dir = os.path.join(data_dir, "awg", "waves")
        if not os.path.isdir(wave_dir):
            raise Exception(
                "AWG module wave directory {} does not exist or is not a "
                "directory".format(
                    wave_dir))
        wave_file = os.path.join(wave_dir, wave_name + '.wave')

        wave_array = zhinst.utils.convert_awg_waveform(waveform)
        wave_array.tofile(wave_file)

    def upload_waveforms(self, sequence_names, sequence_channels, sequence_items, reload=True):
        channel_map = {}
        for name, channel, sequence in zip(sequence_names, sequence_channels, sequence_items):
            if len(channel) == 2:
                sequence = sequence.astype(int)
            channel = channel[0] + 1
            logger.info(f'writing wave {name}')
            if self._use_binary_waves:
                self.waveform_to_wave(self.__awg, waveform=sequence, wave_name=name)
            else:
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

    def delete_waveforms(self):
        pass
