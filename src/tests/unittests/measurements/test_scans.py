""" Basic scan functions.

This module contains test functions for basic scans, e.g. scan1D, scan2D, etc.
This is part of qtt.

"""

import sys
import warnings
import random
from unittest import TestCase
from unittest.mock import MagicMock, patch
import tempfile

import numpy as np
import qcodes
from qcodes import Parameter, ManualParameter
from qcodes.instrument_drivers.devices import VoltageDivider
from qcodes.instrument_drivers.ZI.ZIUHFLI import ZIUHFLI

import zhinst

import qtt.algorithms.onedot
import qtt.gui.live_plotting
import qtt.utilities.tools
from qtt.instrument_drivers.virtual_instruments import VirtualIVVI
from qtt.measurements.scans import (get_instrument_parameter, instrumentName,
                                    measure_segment_scope_reader, fastScan,
                                    sample_data_t, scan1D, scan2D, scanjob_t,
                                    get_sampling_frequency)
from qtt.structures import MultiParameter
from qtt.instrument_drivers.simulation_instruments import SimulationDigitizer
from qtt.measurements.scans import measuresegment

sys.modules['pyspcm'] = MagicMock()
from qcodes_contrib_drivers.drivers.Spectrum.M4i import M4i
del sys.modules['pyspcm']


class TestScans(TestCase):

    def setUp(self):
        qcodes.DataSet.default_io = qcodes.DiskIO(tempfile.mkdtemp(prefix='qtt-unittests'))

    def test_get_instrument_parameter(self):
        instrument = VirtualIVVI(instrumentName('test'), None)
        ix, p = get_instrument_parameter((instrument.name, 'dac2'))
        self.assertEqual(id(ix), id(instrument))
        self.assertEqual(id(p), id(instrument.dac2))
        ix, p = get_instrument_parameter((instrument, 'dac2'))
        self.assertEqual(id(p), id(instrument.dac2))
        ix, p = get_instrument_parameter(instrument.name + '.dac2')
        self.assertEqual(id(p), id(instrument.dac2))
        instrument.close()

    def test_sample_data(self):
        s = sample_data_t()
        s['gate_boundaries'] = {'D0': [-500, 100]}
        v = s.restrict_boundaries('D0', 1000)
        self.assertEqual(100, v)

    def test_scan1D(self, verbose=0):
        p = Parameter('p', set_cmd=None)
        q = Parameter('q', set_cmd=None)
        r = VoltageDivider(p, 4)
        _ = MultiParameter(instrumentName('multi_param'), [p, q])

        gates = VirtualIVVI(
            name=qtt.measurements.scans.instrumentName('gates'), model=None)
        station = qcodes.Station(gates)
        station.gates = gates

        if verbose:
            print('test_scan1D: running scan1D')
        scanjob = scanjob_t({'sweepdata': dict(
            {'param': p, 'start': 0, 'end': 10, 'step': 2, 'wait_time': 0.}), 'minstrument': [r]})
        _ = scan1D(station, scanjob, liveplotwindow=False, verbose=0)

        scanjob = scanjob_t({'sweepdata': dict(
            {'param': p, 'start': 0, 'end': 10, 'step': 2, 'wait_time': 0.}), 'minstrument': [q, r]})
        _ = scan1D(station, scanjob, liveplotwindow=False, verbose=0)

        scanjob = scanjob_t({'sweepdata': dict(
            {'param': 'dac1', 'start': 0, 'end': 10, 'step': 2}), 'minstrument': [r]})
        _ = scan1D(station, scanjob, liveplotwindow=False, verbose=0)

        scanjob = scanjob_t({'sweepdata': dict(
            {'param': {'dac1': 1}, 'start': 0, 'range': 10, 'step': 2}), 'minstrument': [r]})
        data = scan1D(station, scanjob, liveplotwindow=False, verbose=0, extra_metadata={'hi': 'world'})
        self.assertTrue('hi' in data.metadata)
        gates.close()

    def test_scan1D_no_gates(self):
        p = Parameter('p', set_cmd=None)
        r = VoltageDivider(p, 4)
        scanjob = scanjob_t({'sweepdata': {'param': p, 'start': 0, 'end': 10, 'step': 2}, 'minstrument': [r]})
        station = qcodes.Station()
        dataset = scan1D(station, scanjob, liveplotwindow=False, verbose=0)
        default_record_label = 'scan1D'
        self.assertTrue(default_record_label in dataset.location)

    def test_scanjob_record_label(self):
        p = Parameter('p', set_cmd=None)
        r = VoltageDivider(p, 4)

        record_label = '123unittest123'
        scanjob = scanjob_t({'sweepdata': dict(
            {'param': p, 'start': 0, 'end': 10, 'step': 2, 'wait_time': 0.}), 'minstrument': [r]})
        scanjob['dataset_label'] = record_label
        station = qcodes.Station()
        dataset = scan1D(station, scanjob, liveplotwindow=False, verbose=0)
        self.assertTrue(dataset.location.endswith(record_label))

    def test_scan2D(self):
        verbose = 0
        p = Parameter('p', set_cmd=None)
        q = Parameter('q', set_cmd=None)
        r = VoltageDivider(p, 4)
        _ = MultiParameter(instrumentName('multi_param'), [p, q])

        gates = VirtualIVVI(
            name=qtt.measurements.scans.instrumentName('gates'), model=None)
        station = qcodes.Station(gates)
        station.gates = gates

        if verbose:
            print('test_scan2D: running scan2D')
        scanjob = scanjob_t({'sweepdata': dict(
            {'param': p, 'start': 0, 'end': 10, 'step': 2}), 'minstrument': [r]})
        scanjob['stepdata'] = dict(
            {'param': q, 'start': 24, 'end': 30, 'step': 1.})
        _ = scan2D(station, scanjob, liveplotwindow=False, verbose=0)

        scanjob = scanjob_t({'sweepdata': dict({'param': {
            'dac1': 1, 'dac2': .1}, 'start': 0, 'range': 10, 'step': 2}), 'minstrument': [r]})
        scanjob['stepdata'] = dict(
            {'param': {'dac2': 1}, 'start': 24, 'range': 6, 'end': np.NaN, 'step': 1.})
        _ = scan2D(station, scanjob, liveplotwindow=False, verbose=0)

        scanjob = scanjob_t({'sweepdata': dict(
            {'param': {'dac1': 1}, 'start': 0, 'range': 10, 'step': 2}), 'minstrument': [r]})
        scanjob['stepdata'] = {'param': MultiParameter('multi_param', [gates.dac2, gates.dac3])}
        scanjob['stepvalues'] = np.array([[2 * i, 3 * i] for i in range(10)])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                data = scan2D(station, scanjob, liveplotwindow=False, verbose=0)
        except Exception as ex:
            print(ex)
            warnings.warn('MultiParameter test failed!')
        # Test combination of Parameter and vector argument not supported:
        scanjob = scanjob_t({'sweepdata': dict({'param': {
            'dac1': 1}, 'start': 0, 'range': 10, 'step': 2, 'wait_time': 0.}), 'minstrument': [r]})
        scanjob['stepdata'] = dict(
            {'param': q, 'start': 24, 'range': 6, 'end': np.NaN, 'step': 1.})

        self.assertRaises(Exception, scan2D, station, scanjob, liveplotwindow=False, verbose=0)
        gates.close()

    def test_measure_segment_scope_reader_2D(self):
        period = 1e-3
        width = [0.9, 0.9]
        resolution = [96, 96]
        sample_rate = 1e6
        mock_data_arrays = MagicMock(T=[2, 3, 4])

        mock_scope = MagicMock(sample_rate=sample_rate)
        mock_scope.acquire.return_value = mock_data_arrays

        raw_data_mock = MagicMock()
        data_mock = MagicMock()
        waveform = dict(period=period, width_horz=width[0], width_vert=width[1], resolution=resolution)
        average_count = 123

        with patch('qtt.measurements.scans.process_2d_sawtooth', return_value=(data_mock, None)) as process_mock:
            with patch('numpy.array') as array_mock:
                array_mock.return_value.T = raw_data_mock
                result_data = measure_segment_scope_reader(mock_scope, waveform, average_count)

                array_mock.assert_called_once_with(mock_data_arrays)
                mock_scope.acquire.assert_called_once_with(average_count)
                process_mock.assert_called_with(raw_data_mock, period, sample_rate,
                                                resolution, width, fig=None, start_zero=False)
                self.assertEqual(data_mock, result_data)

    def test_measure_segment_scope_reader_1D(self):
        period = 1e-3
        width = 0.9
        sample_rate = 1e6
        mock_data_arrays = MagicMock(T=[2, 3, 4])

        mock_scope = MagicMock(sample_rate=sample_rate)
        mock_scope.acquire.return_value = mock_data_arrays

        raw_data_mock = MagicMock()
        data_mock = MagicMock()
        waveform = dict(period=period, width=width)
        average_count = 123

        with patch('qtt.measurements.scans.process_1d_sawtooth', return_value=(data_mock, None)) as process_mock:
            with patch('numpy.array') as array_mock:
                array_mock.return_value.T = raw_data_mock
                result_data = measure_segment_scope_reader(mock_scope, waveform, average_count)

                array_mock.assert_called_once_with(mock_data_arrays)
                mock_scope.acquire.assert_called_once_with(average_count)
                process_mock.assert_called_with(raw_data_mock, [width], period, sample_rate,
                                                resolution=None, start_zero=False, fig=None)
                self.assertEqual(data_mock, result_data)

    def test_fastScan_no_awg(self):
        station = MagicMock()
        station.awg = None
        station.virtual_awg = None
        scanjob = scanjob_t({'sweepdata': dict({'param': {'dac1': 1},
                                                'start': 0,
                                                'range': 10,
                                                'step': 2}),
                             'minstrument': []})

        self.assertEqual(fastScan(scanjob, station), 0)

    def test_measure_segment_scope_reader_no_processing(self):
        mock_scope = MagicMock()
        mock_scope.acquire.return_value = MagicMock()

        average_count = 123
        numpy_array_mock = MagicMock()
        with patch('numpy.array', return_value=numpy_array_mock):
            result_data = measure_segment_scope_reader(mock_scope, {}, average_count, process=False)
            mock_scope.acquire.assert_called_once_with(average_count)
            self.assertEqual(numpy_array_mock, result_data)

    @staticmethod
    def test_measure_segment_m4i_has_correct_output():
        expected_data = np.array([1, 2, 3, 4])
        waveform = {'bla': 1, 'ble': 2, 'blu': 3}
        number_of_averages = 100
        read_channels = [0, 1]

        with patch('qtt.measurements.scans.measuresegment_m4i') as measure_segment_mock:

            m4i_digitizer = M4i('test')
            measure_segment_mock.return_value = expected_data

            actual_data = measuresegment(waveform, number_of_averages, m4i_digitizer, read_channels)
            np.testing.assert_array_equal(actual_data, expected_data)
            measure_segment_mock.assert_called_with(m4i_digitizer, waveform, read_channels,
                                                    2000, number_of_averages, process=True)

            m4i_digitizer.close()

    @staticmethod
    def test_measure_segment_uhfli_has_correct_output():
        expected_data = np.array([1, 2, 3, 4])
        waveform = {'bla': 1, 'ble': 2, 'blu': 3}
        number_of_averages = 100
        read_channels = [0, 1]

        with patch.object(zhinst.utils, 'create_api_session', return_value=3 * (MagicMock(),)), \
                patch('qtt.measurements.scans.measure_segment_uhfli') as measure_segment_mock:

            uhfli_digitizer = ZIUHFLI('test', 'dev1234')
            measure_segment_mock.return_value = expected_data

            actual_data = measuresegment(waveform, number_of_averages, uhfli_digitizer, read_channels)
            np.testing.assert_array_equal(actual_data, expected_data)
            measure_segment_mock.assert_called_with(uhfli_digitizer, waveform, read_channels, number_of_averages)

            uhfli_digitizer.close()

    @staticmethod
    def test_measure_segment_simulator_has_correct_output():
        expected_data = np.array([1, 2, 3, 4])
        waveform = {'bla': 1, 'ble': 2, 'blu': 3}
        number_of_averages = 100
        read_channels = [0, 1]

        with patch('qtt.instrument_drivers.simulation_instruments.SimulationDigitizer',
                   spec=SimulationDigitizer) as simulation_digitizer:

            simulation_digitizer.measuresegment.return_value = expected_data
            actual_data = measuresegment(waveform, number_of_averages, simulation_digitizer, read_channels)
            np.testing.assert_array_equal(actual_data, expected_data)
            simulation_digitizer.measuresegment.assert_called_with(waveform, channels=read_channels)

    def test_measure_segment_invalid_device(self):
        waveform = {'bla': 1, 'ble': 2, 'blu': 3}
        read_channels = [0, 1]

        self.assertRaises(Exception, measuresegment, waveform, 100, MagicMock(), read_channels)

    @staticmethod
    def test_measure_segment_no_data_raises_warning():
        expected_data = np.array([])
        waveform = {'bla': 1, 'ble': 2, 'blu': 3}
        number_of_averages = 100
        read_channels = [0, 1]

        with patch('qtt.instrument_drivers.simulation_instruments.SimulationDigitizer',
                   spec=SimulationDigitizer) as simulation_digitizer, patch('warnings.warn') as warn_mock:

            simulation_digitizer.measuresegment.return_value = expected_data
            actual_data = measuresegment(waveform, number_of_averages, simulation_digitizer, read_channels)
            warn_mock.assert_called_once_with('measuresegment: received empty data array')
            np.testing.assert_array_equal(expected_data, actual_data)

    def test_get_sampling_frequency_m4i(self):
        expected_value = 12.345e6

        m4i_digitizer = M4i('test')
        m4i_digitizer.sample_rate = ManualParameter('sample_rate', initial_value=expected_value)

        actual_value = get_sampling_frequency(m4i_digitizer)
        self.assertEqual(expected_value, actual_value)

        m4i_digitizer.close()

    def test_convert_scanjob_vec_scan1Dfast(self):
        p = Parameter('p', set_cmd=None)
        gates = VirtualIVVI(
            name=qtt.measurements.scans.instrumentName('gates'), model=None)
        station = qcodes.Station(gates)
        station.gates = gates

        scanjob = scanjob_t({'scantype': 'scan1Dfast', 'sweepdata': {'param': p, 'start': -2., 'end': 2., 'step': .4}})
        _, sweepvalues = scanjob._convert_scanjob_vec(station)
        actual_values = sweepvalues._values
        expected_values = [-2.0, -1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
        for a_val, e_val in zip(actual_values, expected_values):
            self.assertAlmostEqual(a_val, e_val, 12)
        self.assertEqual(sweepvalues._value_snapshot[0]['num'], 11)
        gates.close()

    def test_convert_scanjob_vec_scan1Dfast_range(self):
        gates = VirtualIVVI(
            name=qtt.measurements.scans.instrumentName('gates'), model=None)
        station = qcodes.Station(gates)
        station.gates = gates

        scanjob = scanjob_t({'scantype': 'scan1Dfast', 'sweepdata': {'param': 'dac1', 'range': 8, 'step': 2}})
        _, sweepvalues = scanjob._convert_scanjob_vec(station)
        actual_values = sweepvalues._values
        expected_values = [-4.0, -2.0, 0.0, 2.0, 4.0]
        for a_val, e_val in zip(actual_values, expected_values):
            self.assertAlmostEqual(a_val, e_val, 12)
        self.assertEqual(sweepvalues._value_snapshot[0]['num'], 5)
        gates.close()

    def test_convert_scanjob_vec_scan1Dfast_adjust_sweeplength(self):
        p = Parameter('p', set_cmd=None)
        gates = VirtualIVVI(
            name=qtt.measurements.scans.instrumentName('gates'), model=None)
        station = qcodes.Station(gates)
        station.gates = gates

        scanjob = scanjob_t({'scantype': 'scan1Dfast', 'sweepdata': {'param': p, 'start': -2, 'end': 2, 'step': .4}})
        _, sweepvalues = scanjob._convert_scanjob_vec(station, sweeplength=5)
        actual_values = sweepvalues._values
        expected_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
        self.assertEqual(expected_values, actual_values)
        self.assertEqual(sweepvalues._value_snapshot[0]['num'], 5)
        gates.close()

    def test_convert_scanjob_vec_scan1Dfast_adjust_sweeplength_adjusted_end(self):
        p = Parameter('p', set_cmd=None)
        gates = VirtualIVVI(
            name=qtt.measurements.scans.instrumentName('gates'), model=None)
        station = qcodes.Station(gates)
        station.gates = gates

        scanjob = scanjob_t({'scantype': 'scan1Dfast',
                             'sweepdata': {'param': p, 'start': -20, 'end': 20, 'step': .0075}})
        _, sweepvalues = scanjob._convert_scanjob_vec(station)
        actual_values = sweepvalues._values
        self.assertEqual(actual_values[0], -20.0)
        self.assertAlmostEqual(scanjob['sweepdata']['end'], 20.0 - 0.0025, 10)
        self.assertEqual(sweepvalues._value_snapshot[0]['num'], 5334)

        scanjob = scanjob_t({'scantype': 'scan1Dfast',
                             'sweepdata': dict({'param': p, 'start': -500, 'end': 1,
                                                'step': .8, 'wait_time': 3e-3})})
        _, sweepvalues = scanjob._convert_scanjob_vec(station)
        actual_values = sweepvalues._values
        self.assertEqual(actual_values[0], -500.0)
        self.assertAlmostEqual(scanjob['sweepdata']['end'], 1 - 0.2, 10)
        self.assertEqual(sweepvalues._value_snapshot[0]['num'], 627)

        gates.close()

    def test_convert_scanjob_vec_adjust_values_randomly_will_never_raise_exception(self):
        p = Parameter('p', set_cmd=None)
        gates = VirtualIVVI(
            name=qtt.measurements.scans.instrumentName('gates'), model=None)
        station = qcodes.Station(gates)
        station.gates = gates

        idx = 1
        for idx in range(1, 100):
            start = -random.randint(1, 20)
            end = random.randint(1, 20)
            step = random.randint(1, 40) / (idx * 10)
            scanjob = scanjob_t({'scantype': 'scan1Dfast',
                                 'sweepdata': {'param': p, 'start': start, 'end': end, 'step': step}})
            _, sweepvalues = scanjob._convert_scanjob_vec(station)

        # all the conversions were successful
        self.assertEqual(idx, 99)
        gates.close()

    def test_convert_scanjob_vec_scan2Dfast(self):
        p = Parameter('p', set_cmd=None)
        q = Parameter('q', set_cmd=None)
        r = VoltageDivider(p, 4)
        _ = MultiParameter(instrumentName('multi_param'), [p, q])

        gates = VirtualIVVI(
            name=qtt.measurements.scans.instrumentName('gates'), model=None)
        station = qcodes.Station(gates)
        station.gates = gates

        scanjob = scanjob_t({'scantype': 'scan2Dfast',
                             'sweepdata': dict(
                                 {'param': p, 'start': 0, 'end': 10, 'step': 4}), 'minstrument': [r]})
        scanjob['stepdata'] = dict(
            {'param': q, 'start': 24, 'end': 32, 'step': 1.})

        stepvalues, sweepvalues = scanjob._convert_scanjob_vec(station, 3, 5)
        actual_stepvalues = stepvalues._values
        expected_stepvalues = [24.0, 28.0, 32.0]
        self.assertEqual(expected_stepvalues, actual_stepvalues)
        self.assertEqual(stepvalues._value_snapshot[0]['num'], 3)

        actual_sweepvalues = sweepvalues._values
        expected_sweepvalues = [0, 2.5, 5.0, 7.5, 10.0]
        self.assertEqual(expected_sweepvalues, actual_sweepvalues)
        self.assertEqual(sweepvalues._value_snapshot[0]['num'], 5)
        gates.close()
