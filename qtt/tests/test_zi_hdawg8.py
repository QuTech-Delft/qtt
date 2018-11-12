import unittest
from unittest.mock import MagicMock, call

from qtt.instrument_drivers.virtualAwg.awgs.ZurichInstrumentsHDAWG8 import ZurichInstrumentsHDAWG8
from qtt.instrument_drivers.virtualAwg.awgs.common import AwgCommonError


class TestZurichInstrumentsHDAWG8(unittest.TestCase):
    def setUp(self):
        class ZIHDAWG8(MagicMock):
            pass

        self.awg = ZIHDAWG8()
        self.zi_hdawg8 = ZurichInstrumentsHDAWG8(self.awg, 0)

    def test_enable_outputs(self):
        self.zi_hdawg8.enable_outputs()
        calls = [call.enable_channel(ch) for ch in range(0, 8)]
        self.awg.assert_has_calls(calls)

        with self.assertRaises(AwgCommonError):
            self.zi_hdawg8.enable_outputs([0, 1, 2, 3, 8])

        self.zi_hdawg8.enable_outputs([6, 7])
        calls = [call.enable_channel(ch) for ch in range(6, 7)]
        self.awg.assert_has_calls(calls)

    def test_disable_outputs(self):
        self.zi_hdawg8.disable_outputs()
        calls = [call.disable_channel(ch) for ch in range(0, 8)]
        self.awg.assert_has_calls(calls)

        with self.assertRaises(AwgCommonError):
            self.zi_hdawg8.disable_outputs([0, 1, 2, 3, 8])

        self.zi_hdawg8.disable_outputs([6, 7])
        calls = [call.disable_channel(ch) for ch in range(6, 7)]
        self.awg.assert_has_calls(calls)

    def test_change_setting(self):
        self.awg.get.return_value = 0
        self.zi_hdawg8.change_setting('sampling_rate', 2.4e9)
        self.assertEqual(self.zi_hdawg8.retrieve_setting('sampling_rate'), 2.4e9)

        with self.assertRaises(ValueError):
            self.zi_hdawg8.change_setting('gain', 0.5)

    def test_update_sampling_rate(self):
        sample_rates = [2400000000.0, 1200000000.0, 600000000.0, 300000000.0, 150000000.0, 72000000.0, 37500000.0,
                        18750000.0, 9400000.0, 4500000.0, 2340000.0, 1200.0, 586000.0, 293000.0]

        for sample_rate in sample_rates:
            self.zi_hdawg8.update_sampling_rate(sample_rate)
        calls = [call.set('awgs_0_time', i) for i in range(0, 14)]
        self.awg.assert_has_calls(calls)

        with self.assertRaises(ValueError):
            self.zi_hdawg8.update_sampling_rate(99)

    def test_retrieve_sampling_rate(self):
        sampling_rate_index = 5
        self.awg.get.return_value = sampling_rate_index
        self.assertEqual(72e6, self.zi_hdawg8.retrieve_sampling_rate())

    def test_update_gain(self):
        self.zi_hdawg8.update_gain(0.5)
        calls = [call.set('sigouts_{}_range'.format(ch), 0.5) for ch in range(8)]
        self.awg.assert_has_calls(calls)

    def test_retrieve_gain(self):
        self.awg.get.return_value = 0.2
        self.assertEqual(0.2, self.zi_hdawg8.retrieve_gain())

        with self.assertRaises(ValueError):
            self.awg.get.side_effect = lambda v: v
            self.zi_hdawg8.retrieve_gain()

    def test_upload_waveforms(self):
        sequence_names = ['seq1', 'seq2', 'seq3']
        sequence_channels = [(1, 1), (1, 0, 1), (2, 0)]
        sequence_items = [range(10), range(1, 11), range(2, 12)]
        self.awg.generate_csv_sequence_program.return_value = 'program'
        self.zi_hdawg8.upload_waveforms(sequence_names, sequence_channels, sequence_items)
        calls = [call.waveform_to_csv('seq1', range(10)),
                 call.waveform_to_csv('seq2', range(1, 11)),
                 call.waveform_to_csv('seq3', range(2, 12)),
                 call.generate_csv_sequence_program(sequence_names, [2, 2, 3]),
                 call.upload_sequence_program(0, 'program')]
        self.awg.assert_has_calls(calls)
