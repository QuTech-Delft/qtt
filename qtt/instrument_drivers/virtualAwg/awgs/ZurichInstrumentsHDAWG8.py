import os
import time

from itertools import groupby
from operator import itemgetter

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
        self.__awg_module = awg._dev.awgModule
        self.__daq = awg._dev.daq
        self.__awg = awg

    @property
    def fetch_awg(self):
        return self.__awg

    def run(self):
        command = "/{}/awgs/*/enable".format(self.__device_name)
        self.__daq.setInt(command, 1)

    def stop(self):
        command = "/{}/awgs/*/enable".format(self.__device_name)
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

    def retrieve_sampling_rate(self):
        #ziDAQ('setInt', '/dev8044/awgs/0/time', 0);
        return 2.4e9

    def update_gain(self, gain):
        raise NotImplementedError

    def retrieve_gain(self):
        # ziDAQ('setDouble', '/dev8044/sigouts/0/range', 1);
        return 1

    def upload_waveforms(self, sequence_names, sequence_channels, sequence_items, reload=True):
        waves = list()
        sequence_header = ''
        for (channel, name, sequence) in zip(sequence_channels, sequence_names, sequence_items):
            waves.append((channel[0], name))
            if 'marker' in name:
                index = 0
                marker_wave_names = list()
                marker_pos = [(len(list(items)), value) for value, items in groupby(sequence)]
                for count, value in marker_pos:
                    marker_name = '{}_{}'.format(name, index)
                    marker_wave_names.append(marker_name)
                    sequence_header += 'wave {} = marker({}, {});\n'.format(marker_name, count, 0 if value == 0 else 1)
                    index += 1
                sequence_header += 'wave {} = join({});\n'.format(name, ', '.join(marker_wave_names))
            else:
                file_name = name + '.csv'
                source_data = '\n'.join(str(item) for item in sequence)
                self._upload_waveform(source_data, file_name)
                sequence_header += 'wave {} = "{}";\n'.format(name, name)
        wave_items = [(k, list(list(zip(*g))[1])) for k, g in groupby(waves, itemgetter(0))]
        sequence_while = '\nwhile(1){\n'
        for channel, wave_names in wave_items:
            play_wave_name = '_'.join(wave_names)
            if len(wave_names) > 1:
                sequence_header += 'wave {} = {};\n'.format(play_wave_name, ' + '.join(wave_names))
            sequence_while += 'playWave({}, {});\n'.format(channel, play_wave_name)
        source_file = sequence_header + sequence_while + '}\n'
        self._upload_sequence(source_file)

    def _upload_waveform(self, source, file_name='wave_form.csv'):
        data_dir = self.__awg_module.getString('awgModule/directory')
        src_dir = os.path.join(data_dir, "awg", "waves")
        if not os.path.isdir(src_dir):
            raise Exception("AWG module wave directory {} does not exist or is not a directory".format(src_dir))
        with open(os.path.join(src_dir, file_name), "w") as f:
            f.write(source)

    def _upload_sequence(self, source, file_name='source_file.seqc'):
        data_dir = self.__awg_module.getString('awgModule/directory')
        src_dir = os.path.join(data_dir, "awg", "src")
        if not os.path.isdir(src_dir):
            raise Exception("AWG module wave directory {} does not exist or is not a directory".format(src_dir))

        with open(os.path.join(src_dir, file_name), "w") as f:
            f.write(source)

        timeout = 20
        self.__awg_module.set('awgModule/compiler/sourcefile', file_name)
        self.__awg_module.set('awgModule/compiler/start', 1)
        t0 = time.time()
        while self.__awg_module.getInt('awgModule/compiler/status') == -1:
            time.sleep(0.1)
            if time.time() - t0 > timeout:
                Exception("Timeout")

        status = self.__awg_module.getInt('awgModule/compiler/status')
        if status == 1:
            compiler_error = self.__awg_module.getString('awgModule/compiler/statusstring')
            raise Exception(compiler_error)
        elif status == 2:
            compiler_warning = self.__awg_module.getString('awgModule/compiler/statusstring')
            print("Compilation successful with warnings, will upload the program to the instrument.")
            print("Compiler warning: ", compiler_warning)
        elif status != 0:
            raise Exception('Other error...')
        i = 0
        time.sleep(0.2)
        while self.__awg_module.getDouble('awgModule/progress') < 1.0:
            time.sleep(0.5)
            i += 1

    def retrieve_waveforms(self):
        raise NotImplementedError

    def delete_waveforms(self):
        pass

