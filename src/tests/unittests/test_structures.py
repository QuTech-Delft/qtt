import unittest
import pickle
import json
import tempfile

import qcodes
import qcodes.data.io
from qcodes.data.data_set import DataSet
import qtt.simulation.virtual_dot_array
from qtt.structures import onedot_t, MultiParameter, CombiParameter


class TestStructures(unittest.TestCase):

    def setUp(self):
        DataSet.default_io = qcodes.data.io.DiskIO(tempfile.mkdtemp(prefix='qtt-unittests'))

    def test_spin_structures(self, verbose=0):
        o = onedot_t('dot1', ['L', 'P1', 'D1'], station=None)
        if verbose:
            print('test_spin_structures: %s' % (o,))
        _ = pickle.dumps(o)
        x = json.dumps(o)
        if verbose:
            print('test_spin_structures: %s' % (x,))
        self.assertTrue('"gates": "dot1"' in x)
        self.assertTrue('"name": ["L", "P1", "D1"]' in x)
        self.assertTrue('"transport_instrument": null' in x)
        self.assertTrue('"instrument": null' in x)

    @staticmethod
    def test_sensingdot_t(fig=None, verbose=0):
        station = qtt.simulation.virtual_dot_array.initialize(verbose=verbose)
        sensing_dot = qtt.structures.sensingdot_t(
            ['SD1a', 'SD1b', 'SD1c'], station=station, minstrument='keithley1.amplitude')
        sensing_dot.verbose = verbose
        if fig is not None:
            _ = sensing_dot.autoTune(step=-8, fig=fig)
        qtt.simulation.virtual_dot_array.close(verbose=verbose)

    def test_multi_parameter(self):
        p = qcodes.Parameter('p', set_cmd=None)
        q = qcodes.Parameter('q', set_cmd=None)
        mp = MultiParameter('multi_param', [p, q])
        mp.set([1, 2])
        _ = mp.get()
        mp = CombiParameter('multi_param', [p, q])
        mp.set([1, 3])
        v = mp.get()
        self.assertTrue(v == 2)
