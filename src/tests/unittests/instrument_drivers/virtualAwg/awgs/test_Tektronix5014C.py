from unittest import TestCase
from unittest.mock import MagicMock, call

import numpy as np

from qtt.instrument_drivers.virtualAwg.awgs.common import AwgCommonError
from qtt.instrument_drivers.virtualAwg.awgs.Tektronix5014C import Tektronix5014C_AWG


class TestTektronix5014C(TestCase):

    def setUp(self):
        class Tektronix5014C(MagicMock):
            pass

        class Tektronix5014C_AWG_non_protected(Tektronix5014C_AWG):

            def set_sequence(self, channels, sequence):
                return self._set_sequence(channels, sequence)

        self.awg = Tektronix5014C()
        type(self.awg).__name__ = 'Tektronix_AWG5014'
        self.awg_backend = Tektronix5014C_AWG_non_protected(self.awg)

    def test_initialize_raises_awg_error(self):
        awg = MagicMock(name='fake')
        self.assertRaises(ValueError, Tektronix5014C_AWG, awg)

    def test_fetch_awg(self):
        self.assertEqual(self.awg, self.awg_backend.fetch_awg)

    def test_run(self):
        self.awg_backend.run()
        self.awg.run.assert_called_once()

    def test_stop(self):
        self.awg_backend.stop()
        self.awg.stop.assert_called_once()

    def test_reset(self):
        self.awg_backend.reset()
        self.awg.reset.assert_called_once()

    def test_enable_outputs_no_arguments(self):
        self.awg_backend.enable_outputs()
        calls = [call(f'ch{channel}_state', 1) for channel in range(1, 5)]
        self.awg.set.assert_has_calls(calls)

    def test_enable_outputs_with_arguments(self):
        channels = [1, 2]
        self.awg_backend.enable_outputs(channels)
        calls = [call(f'ch{channel}_state', 1) for channel in range(1, 3)]
        self.awg.set.assert_has_calls(calls)
        for channel in range(3, 5):
            self.assertTrue(call(f'ch{channel}_state', 1) not in self.awg.set.call_args_list)

    def test_enable_outputs_raises_error(self):
        invalid_channel = [666]
        with self.assertRaisesRegex(ValueError, 'Invalid channel numbers'):
            self.awg_backend.enable_outputs(invalid_channel)

    def test_disable_outputs_no_argument(self):
        self.awg_backend.disable_outputs()
        calls = [call(f'ch{channel}_state', 0) for channel in range(1, 5)]
        self.awg.set.assert_has_calls(calls)

    def test_disable_outputs_with_arguments(self):
        channels = [1, 2]
        self.awg_backend.disable_outputs(channels)
        calls = [call(f'ch{channel}_state', 0) for channel in range(1, 3)]
        self.awg.set.assert_has_calls(calls)
        for channel in range(3, 5):
            self.assertTrue(call(f'ch{channel}_state', 0) not in self.awg.set.call_args_list)

    def test_disable_outputs_raises_error(self):
        invalid_channel = [666]
        with self.assertRaisesRegex(ValueError, 'Invalid channel numbers'):
            self.awg_backend.disable_outputs(invalid_channel)

    def test_change_setting_getter_and_setter(self):
        settings = {
            'marker_low': -0.5,
            'marker_high': 0.5,
            'amplitudes': 2.5,
            'offset': 2.0
        }

        for name, value in settings.items():
            self.awg_backend.change_setting(name, value)
            actual_value = self.awg_backend.retrieve_setting(name)
            self.assertEqual(actual_value, value)

    def test_update_running_mode(self):
        mode = 'SEQ'
        self.awg_backend.update_running_mode(mode)
        self.awg.set.assert_called_once_with('run_mode', mode)

    def test_update_running_mode_raises_error(self):
        mode = 'WRONG'
        with self.assertRaisesRegex(ValueError, 'Invalid AWG mode'):
            self.awg_backend.update_running_mode(mode)

    def test_retrieve_running_mode(self):
        mode = 'CONT'
        self.awg.get.return_value = mode
        actual_mode = self.awg_backend.retrieve_running_mode()
        self.assertEqual(mode, actual_mode)

    def test_update_sample_rate(self):
        sample_rate = 1.2e9
        self.awg_backend.update_sampling_rate(sample_rate)
        self.awg.set.assert_called_once_with('clock_freq', sample_rate)

    def test_retrieve_sample_rate(self):
        sample_rate = 0.8e9
        self.awg.get.return_value = sample_rate
        actual_sample_rate = self.awg_backend.retrieve_sampling_rate()
        self.assertEqual(sample_rate, actual_sample_rate)

    def test_update_gain(self):
        gain = 1.2
        self.awg_backend.update_gain(gain)
        calls = [call(f'ch{channel}_amp', gain) for channel in range(1, 4)]
        self.awg.set.assert_has_calls(calls)

    def test_retrieve_gain(self):
        gain = 0.8
        self.awg.get.return_value = gain
        actual_gain = self.awg_backend.retrieve_gain()
        self.assertEqual(gain, actual_gain)

    def test_retrieve_gain_raises_error(self):
        def gain_side_effect(value):
            return 1.0 if value == 'ch1_amp' else 2.0

        self.awg.get = gain_side_effect
        self.assertRaises(AwgCommonError, self.awg_backend.retrieve_gain)

    def test_upload_waveforms(self):
        data_seq1 = np.array(range(10, 20), dtype=float)
        data_seq2 = np.array(range(20, 30), dtype=float)
        data_mark = np.array(range(30, 40), dtype=float)
        data_zero = np.zeros(10, dtype=float)

        sequence_names = ['wave1', 'wave2', 'mark']
        sequence_channels = [(1, ), (2, ), (1, 1)]
        sequence_items = [data_seq1, data_seq2, data_mark]

        mock_awg_file = 'test_file'
        awg_file_name = 'default.awg'
        mock_awg_directory = 'test_directory'
        self.awg.generate_awg_file.return_value = mock_awg_file
        self.awg.visa_handle.query.return_value = mock_awg_directory

        self.awg_backend.upload_waveforms(sequence_names, sequence_channels, sequence_items)

        upload_data = self.awg.pack_waveform.call_args_list
        np.testing.assert_array_equal(upload_data[0][0], (data_seq1, data_mark, data_zero))
        np.testing.assert_array_equal(upload_data[1][0], (data_seq2, data_zero, data_zero))
        self.awg.visa_handle.write.assert_called_once_with('MMEMory:CDIRectory "C:\\Users\\OEM\\Documents"')
        self.awg.send_awg_file.assert_called_once_with(awg_file_name, mock_awg_file)
        self.awg.load_awg_file.assert_called_once_with(f'{mock_awg_directory}{awg_file_name}')
        self.awg.set_sqel_goto_state.assert_called_once_with(1, 1)

    def test_create_waveform_data(self):
        data_seq1 = np.array(range(10), dtype=float)
        data_mark = np.array(range(1, 11), dtype=float)
        data_seq2 = np.array(range(2, 12), dtype=float)
        data_zero = np.zeros(10, dtype=float)
        names = ['seq1', 'mark', 'seq2']
        channels = [(1, 0), (1, 1), (2, 0)]
        items = [data_seq1, data_mark, data_seq2]

        channel_data, waveform_data = Tektronix5014C_AWG.create_waveform_data(names, channels, items)

        np.testing.assert_array_equal(waveform_data['seq1'][0], data_seq1)
        np.testing.assert_array_equal(waveform_data['seq1'][1], data_mark)
        np.testing.assert_array_equal(waveform_data['seq1'][2], data_zero)
        np.testing.assert_array_equal(waveform_data['seq2'][0], data_seq2)
        np.testing.assert_array_equal(waveform_data['seq2'][1], data_zero)
        np.testing.assert_array_equal(waveform_data['seq2'][2], data_zero)
        expected_channel_data = {1: ['seq1'], 2: ['seq2']}
        self.assertDictEqual(channel_data, expected_channel_data)

    def test_delete_waveforms(self):
        self.awg_backend.delete_waveforms()
        self.awg.delete_all_waveforms_from_list.assert_called_once()

    def test_delete_sequence(self):
        self.awg_backend.delete_sequence()
        self.awg.write.assert_called_once_with('SEQuence:LENGth 0')

    def test_set_sequence_length(self):
        row_count = 24
        self.awg_backend.set_sequence_length(row_count)
        self.awg.write.assert_called_once_with(f'SEQuence:LENGth {row_count}')

    def test_get_sequence_length(self):
        row_count = 42
        self.awg.ask.return_value = row_count
        actual_row_count = self.awg_backend.get_sequence_length()
        self.assertEqual(row_count, actual_row_count)

    def test_set_sequence_invalid_channel_length(self):
        channels = [1, 2, 3]
        sequence = [['fake']]
        self.assertRaisesRegex(ValueError, 'Invalid sequence and channel count!',
                               self.awg_backend.set_sequence, channels, sequence)

    def test_set_sequence_invalid_sequence_waveform_items(self):
        channels = [1, 2]
        sequence = [['waveform_1', 'waveform_2'], ['waveform_3']]
        self.assertRaisesRegex(ValueError, 'Invalid sequence list lengths!',
                               self.awg_backend.set_sequence, channels, sequence)

    def test_set_sequence_set_sequence_length_called(self):
        row_count = 42
        channels = [1, 2]
        sequence = [['waveform_1', 'waveform_2'], ['waveform_2', 'waveform_1']]
        self.awg.ask.return_value = row_count
        self.awg_backend.set_sequence(channels, sequence)
        self.awg.write.assert_called_once_with(f'SEQuence:LENGth {len(sequence)}')
