import numpy as np
import unittest
import warnings
from unittest.mock import patch
from qtt.instrument_drivers.virtualAwg.templates import DataTypes
from qtt.instrument_drivers.virtualAwg.sequencer import Sequencer
# warnings.filterwarnings('ignore', category=UserWarning, message="gmpy2 not found.")
from qupulse.pulses.plotting import (PlottingNotPossibleException, render)


class TestSequencer(unittest.TestCase):
    def test_make_marker_no_regression(self):
        period = 10e-9
        offset = 0
        uptime = 4e-9
        marker = Sequencer.make_marker(period=period, offset=offset, uptime=uptime)

        self.assertEqual(offset, marker['offset'])
        self.assertEqual(uptime, marker['uptime'])

        parameters = marker['wave'].subtemplates[0].parameter_mapping
        self.assertEqual(10, parameters['period'])
        self.assertEqual(0, parameters['offset'])
        self.assertEqual(4, parameters['uptime'])

    def test_make_marker_no_regression_with_offset(self):
        period = 10e-9
        offset = 3e-9
        uptime = 4e-9
        marker = Sequencer.make_marker(period=period, offset=offset, uptime=uptime)
        self.assertEqual(offset, marker['offset'])
        self.assertEqual(uptime, marker['uptime'])

        parameters = marker['wave'].subtemplates[0].parameter_mapping
        self.assertEqual(3, parameters['offset'])
        self.assertEqual(4, parameters['uptime'])
        self.assertEqual(10, parameters['period'])

    def test_make_marker_negative_offset(self):
        period = 10e-9
        offset = -3e-9
        uptime = 2e-9
        marker = Sequencer.make_marker(period=period, offset=offset, uptime=uptime)
        self.assertEqual(offset, marker['offset'])
        self.assertEqual(uptime, marker['uptime'])

        parameters = marker['wave'].subtemplates[0].parameter_mapping
        self.assertEqual(7, parameters['offset'])
        self.assertEqual(2, parameters['uptime'])
        self.assertEqual(10, parameters['period'])

    def test_make_marker_negative_offset_rollover(self):
        period = 10e-9
        offset = -2e-9
        uptime = 4e-9
        with patch('warnings.warn') as warn:
            marker = Sequencer.make_marker(period=period, offset=offset, uptime=uptime)
            warn.assert_any_call('Marker rolls over to subsequent period.')
        self.assertEqual(offset, marker['offset'])
        self.assertEqual(uptime, marker['uptime'])

        parameters = marker['wave'].subtemplates[0].parameter_mapping
        self.assertEqual(8, parameters['offset'])
        self.assertEqual(4, parameters['uptime'])
        self.assertEqual(10, parameters['period'])

    def test_offset_raises_errors(self):
        period = 10e-9
        offset = -11e-9
        uptime = 2e-9
        with self.assertRaises(ValueError) as error:
            Sequencer.make_marker(period=period, offset=offset, uptime=uptime)
        self.assertIn('Invalid argument value for offset: |-1.1e-08| > 1e-08!', error.exception.args)

        offset = -offset
        with self.assertRaises(ValueError) as error:
            Sequencer.make_marker(period=period, offset=offset, uptime=uptime)
        self.assertIn('Invalid argument value for offset: |1.1e-08| > 1e-08!', error.exception.args)

    def test_uptime_raises_errors(self):
        period = 10e-9
        offset = 0
        uptime = 0
        with self.assertRaises(ValueError) as error:
            Sequencer.make_marker(period=period, offset=offset, uptime=uptime)
        self.assertIn("Invalid argument value for uptime '0'!", error.exception.args)

        uptime = 11e-9
        with self.assertRaises(ValueError) as error:
            Sequencer.make_marker(period=period, offset=offset, uptime=uptime)
        self.assertIn("Invalid argument value for uptime '1.1e-08'!", error.exception.args)

        uptime = -1e-9
        with self.assertRaises(ValueError) as error:
            Sequencer.make_marker(period=period, offset=offset, uptime=uptime)
        self.assertIn("Invalid argument value for uptime '-1e-09'!", error.exception.args)

    def test_qupulse_sawtooth_HasCorrectProperties(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="qupulse")
            epsilon = 1e-14
            period = 1e-3
            amplitude = 1.5
            sampling_rate = 1e9
            sequence = Sequencer.make_sawtooth_wave(amplitude, period)
            raw_data = Sequencer.get_data(sequence, sampling_rate)
            self.assertTrue(len(raw_data) == sampling_rate * period + 1)
            self.assertTrue(np.abs(np.min(raw_data) + amplitude / 2) <= epsilon)
            self.assertTrue(np.abs(np.max(raw_data) - amplitude / 2) <= epsilon)

    def test_qupulse_template_to_array_new_style_vs_values_old_style(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="qupulse")
            warnings.filterwarnings("ignore", category=DeprecationWarning, message="InstructionBlock API is deprecated")
            period = 1e-3
            amplitude = 1.5
            sampling_rate = 1e9
            sequence = Sequencer.make_sawtooth_wave(amplitude, period)
            template = sequence['wave']
            channels = template.defined_channels
            loop = template.create_program(parameters=dict(),
                                           measurement_mapping={w: w for w in template.measurement_names},
                                           channel_mapping={ch: ch for ch in channels},
                                           global_transformation=None,
                                           to_single_waveform=set())
            (_, voltages_new, _) = render(loop, sampling_rate / 1e9)

            # the value to compare to are calculated using qupulse 0.4 Sequencer class
            self.assertTrue(1000001 == len(voltages_new['sawtooth']))
            self.assertAlmostEqual(-amplitude/2, np.min(voltages_new['sawtooth']), 12)
            self.assertAlmostEqual(amplitude/2, np.max(voltages_new['sawtooth']), 12)

    def test_raw_wave_HasCorrectProperties(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="qupulse")
            period = 1e-3
            sampling_rate = 1e9
            name = 'test_raw_data'
            sequence = {'name': name, 'wave': [0] * int(period * sampling_rate + 1),
                        'type': DataTypes.RAW_DATA}
            raw_data = Sequencer.get_data(sequence, sampling_rate)
            self.assertTrue(len(raw_data) == sampling_rate * period + 1)
            self.assertTrue(np.min(raw_data) == 0)

    def test_serialize_deserialize_pulse(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="qupulse")
            period = 1e-6
            amplitude = 1.5
            sawtooth = Sequencer.make_sawtooth_wave(amplitude, period)
            serialized_pulse = sawtooth['wave']
            serialized_data = Sequencer.serialize(sawtooth)
            self.assertTrue("qupulse.pulses.sequence_pulse_template.SequencePulseTemplate" in serialized_data)
            self.assertTrue("qupulse.pulses.mapping_pulse_template.MappingPulseTemplate" in serialized_data)
            self.assertTrue("\"amplitude\": 0.75" in serialized_data)
            self.assertTrue("\"period\": 1000.0" in serialized_data)
            self.assertTrue("\"width\": 0.95" in serialized_data)
            self.assertTrue("qupulse.pulses.table_pulse_template.TablePulseTemplate" in serialized_data)
            self.assertTrue("sawtooth" in serialized_data)
            deserialized_pulse = Sequencer.deserialize(serialized_data)
            self.assertEqual(serialized_pulse.subtemplates[0].parameter_mapping['period'],
                             deserialized_pulse.subtemplates[0].parameter_mapping['period'])
            self.assertEqual(serialized_pulse.subtemplates[0].parameter_mapping['amplitude'],
                             deserialized_pulse.subtemplates[0].parameter_mapping['amplitude'])
            self.assertEqual(serialized_pulse.subtemplates[0].parameter_mapping['width'],
                             deserialized_pulse.subtemplates[0].parameter_mapping['width'])

    def test_make_pulse_table(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message="qupulse")
            amplitudes = [1, 2, 3]
            waiting_times = [1e-4, 2e-5, 3e-3]
            sampling_rate = 1e9
            pulse_data = Sequencer.make_pulse_table(amplitudes, waiting_times)
            raw_data = Sequencer.get_data(pulse_data, sampling_rate)
            self.assertTrue(raw_data[0] == amplitudes[0])
            self.assertTrue(raw_data[-1] == amplitudes[-1])
