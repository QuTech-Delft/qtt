import unittest
from unittest.mock import patch

from qtt.instrument_drivers.virtualAwg.sequencer import Sequencer


class TestSequencer(unittest.TestCase):
    def test_make_marker_no_regression(self):
        period = 10e-9
        offset = 0
        uptime = 4e-9
        marker = Sequencer.make_marker(period=period, offset=offset, uptime=uptime)
        self.assertEqual(offset, marker['offset'])
        self.assertEqual(uptime, marker['uptime'])

        parameters = marker['wave'].subtemplates[0].parameter_mapping
        self.assertEqual(0, parameters['rollover'])
        self.assertEqual(0, parameters['rollover_duration'])
        self.assertEqual(0, parameters['pulse_rise'])
        self.assertEqual(4, parameters['pulse_fall'])
        self.assertEqual(10, parameters['period'])

    def test_make_marker_no_regression_with_offset(self):
        period = 10e-9
        offset = 3e-9
        uptime = 4e-9
        marker = Sequencer.make_marker(period=period, offset=offset, uptime=uptime)
        self.assertEqual(offset, marker['offset'])
        self.assertEqual(uptime, marker['uptime'])

        parameters = marker['wave'].subtemplates[0].parameter_mapping
        self.assertEqual(0, parameters['rollover'])
        self.assertEqual(0, parameters['rollover_duration'])
        self.assertEqual(3, parameters['pulse_rise'])
        self.assertEqual(7, parameters['pulse_fall'])
        self.assertEqual(10, parameters['period'])

    def test_make_marker_negative_offset(self):
        period = 10e-9
        offset = -3e-9
        uptime = 2e-9
        marker = Sequencer.make_marker(period=period, offset=offset, uptime=uptime)
        self.assertEqual(offset, marker['offset'])
        self.assertEqual(uptime, marker['uptime'])

        parameters = marker['wave'].subtemplates[0].parameter_mapping
        self.assertEqual(0, parameters['rollover'])
        self.assertEqual(0, parameters['rollover_duration'])
        self.assertEqual(7, parameters['pulse_rise'])
        self.assertEqual(9, parameters['pulse_fall'])
        self.assertEqual(10, parameters['period'])

    def test_make_marker_negative_offset_rollover(self):
        period = 10e-9
        offset = -2e-9
        uptime = 4e-9
        with patch('warnings.warn') as warn:
            marker = Sequencer.make_marker(period=period, offset=offset, uptime=uptime)
            warn.assert_called_once_with('Marker rolls over to subsequent period.')
        self.assertEqual(offset, marker['offset'])
        self.assertEqual(uptime, marker['uptime'])

        parameters = marker['wave'].subtemplates[0].parameter_mapping
        self.assertEqual(1, parameters['rollover'])
        self.assertEqual(2, parameters['rollover_duration'].evaluate_numeric())
        self.assertEqual(8, parameters['pulse_rise'])
        self.assertEqual(10, parameters['pulse_fall'])
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
