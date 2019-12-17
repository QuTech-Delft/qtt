""" Basic scan functions.

This module contains test functions for basic scans, e.g. scan1D, scan2D, etc.
This is part of qtt.

"""

import sys
import warnings
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import qcodes
from qcodes import Parameter
from qcodes.instrument_drivers.devices import VoltageDivider
from qcodes.instrument_drivers.ZI.ZIUHFLI import ZIUHFLI
import zhinst

import qtt.algorithms.onedot
import qtt.gui.live_plotting
import qtt.utilities.tools
from qtt.instrument_drivers.virtual_instruments import VirtualIVVI
from qtt.measurements.scans import (get_instrument_parameter, instrumentName,
                                    measure_segment_scope_reader, fastScan,
                                    sample_data_t, scan1D, scan2D, scanjob_t)
from qtt.structures import MultiParameter
from qtt.instrument_drivers.simulation_instruments import SimulationDigitizer
from qtt.measurements.scans import measuresegment

sys.modules['pyspcm'] = MagicMock()
from qcodes_contrib_drivers.drivers.Spectrum.M4i import M4i
del sys.modules['pyspcm']


class TestScans(TestCase):

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
        station=qcodes.Station()
        dataset = scan1D(station, scanjob, liveplotwindow=False, verbose=0)
        default_record_label = 'scan1D'
        self.assertTrue(default_record_label in dataset.location)

    def test_scanjob_record_label(self):
        p = Parameter('p', set_cmd=None)
        r = VoltageDivider(p, 4)

        record_label='123unittest123'
        scanjob = scanjob_t({'sweepdata': dict(
            {'param': p, 'start': 0, 'end': 10, 'step': 2, 'wait_time': 0.}), 'minstrument': [r]})
        scanjob['dataset_label']=record_label
        station=qcodes.Station()
        dataset = scan1D(station, scanjob, liveplotwindow=False, verbose=0)
        self.assertTrue(dataset.location.endswith(record_label))

    @staticmethod
    def test_scan2D(verbose=0):
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
        # not supported:
        try:
            scanjob = scanjob_t({'sweepdata': dict({'param': {
                'dac1': 1}, 'start': 0, 'range': 10, 'step': 2, 'wait_time': 0.}), 'minstrument': [r]})
            scanjob['stepdata'] = dict(
                {'param': q, 'start': 24, 'range': 6, 'end': np.NaN, 'step': 1.})
            _ = scan2D(station, scanjob, liveplotwindow=False, verbose=0)
        except Exception as ex:
            if verbose:
                print('combination of Parameter and vector argument not supported')
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

        with patch('qtt.measurements.scans.process_2d_sawtooth', return_value = (data_mock, None)) as process_mock:
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

        with patch('qtt.measurements.scans.process_1d_sawtooth', return_value = (data_mock, None)) as process_mock:
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
        station.virtual_awg =  None
        scanjob = scanjob_t({'sweepdata': dict({'param': {'dac1': 1}, 'start': 0, 'range': 10, 'step': 2}), 'minstrument': []})

        self.assertEqual(fastScan(scanjob, station), 0)

    def test_measure_segment_scope_reader_no_processing(self):
        mock_scope = MagicMock()
        mock_scope.acquire.return_value = MagicMock()

        average_count = 123
        numpy_array_mock = MagicMock()
        with patch('numpy.array', return_value=numpy_array_mock):
            result_data = measure_segment_scope_reader(mock_scope, None, average_count, process=False)
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

        with patch('qtt.instrument_drivers.simulation_instruments.SimulationDigitizer', \
                   spec=SimulationDigitizer) as simulation_digitizer, patch('warnings.warn') as warn_mock:

            simulation_digitizer.measuresegment.return_value = expected_data
            actual_data = measuresegment(waveform, number_of_averages, simulation_digitizer, read_channels)
            warn_mock.assert_called_once_with('measuresegment: received empty data array')
            np.testing.assert_array_equal(expected_data, actual_data)
