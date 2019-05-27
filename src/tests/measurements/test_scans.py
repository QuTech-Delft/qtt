""" Basic scan functions.

This module contains test functions for basic scans, e.g. scan1D, scan2D, etc.
This is part of qtt.

"""

import unittest
import numpy as np
import warnings
import qcodes
from qcodes import Parameter
from qcodes.instrument_drivers.devices import VoltageDivider
from qtt.instrument_drivers.virtual_instruments import VirtualIVVI
from qtt.structures import MultiParameter
import qtt.utilities.tools
import qtt.algorithms.onedot
import qtt.gui.live_plotting
from qtt.measurements.scans import scanjob_t, get_instrument_parameter, sample_data_t, instrumentName, scan2D, scan1D


class TestScans(unittest.TestCase):

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

    def test_scan2D(self, verbose=0):
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
        data = scan2D(station, scanjob, liveplotwindow=False, verbose=0)

        scanjob = scanjob_t({'sweepdata': dict({'param': {
            'dac1': 1, 'dac2': .1}, 'start': 0, 'range': 10, 'step': 2}), 'minstrument': [r]})
        scanjob['stepdata'] = dict(
            {'param': {'dac2': 1}, 'start': 24, 'range': 6, 'end': np.NaN, 'step': 1.})
        data = scan2D(station, scanjob, liveplotwindow=False, verbose=0)

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
            data = scan2D(station, scanjob, liveplotwindow=False, verbose=0)
        except Exception as ex:
            if verbose:
                print('combination of Parameter and vector argument not supported')

        if verbose:
            print('test_scan2D: running scan1D')
        scanjob = scanjob_t({'sweepdata': dict(
            {'param': p, 'start': 0, 'end': 10, 'step': 2, 'wait_time': 0.}), 'minstrument': [r]})
        data = scan1D(station, scanjob, liveplotwindow=False, verbose=0)

        scanjob = scanjob_t({'sweepdata': dict(
            {'param': p, 'start': 0, 'end': 10, 'step': 2, 'wait_time': 0.}), 'minstrument': [q, r]})
        data = scan1D(station, scanjob, liveplotwindow=False, verbose=0)

        scanjob = scanjob_t({'sweepdata': dict(
            {'param': 'dac1', 'start': 0, 'end': 10, 'step': 2}), 'minstrument': [r]})
        data = scan1D(station, scanjob, liveplotwindow=False, verbose=0)

        scanjob = scanjob_t({'sweepdata': dict(
            {'param': {'dac1': 1}, 'start': 0, 'range': 10, 'step': 2}), 'minstrument': [r]})
        data = scan1D(station, scanjob, liveplotwindow=False, verbose=0, extra_metadata={'hi': 'world'})
        self.assertTrue('hi' in data.metadata)
        gates.close()
