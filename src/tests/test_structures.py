import unittest
import qcodes
import pickle
import json
import qtt.simulation.virtual_dot_array
from qtt.structures import onedot_t, MultiParameter, CombiParameter


class TestStructures(unittest.TestCase):

    def test_spin_structures(self):
        verbose = 0
        o = onedot_t('dot1', ['L', 'P1', 'D1'], station=None)
        if verbose:
            print('test_spin_structures: %s' % (o,))
        _ = pickle.dumps(o)
        x = json.dumps(o)
        if verbose:
            print('test_spin_structures: %s' % (x,))

    def test_sensingdot_t(self):
        station = qtt.simulation.virtual_dot_array.initialize()
        sensing_dot = qtt.structures.sensingdot_t(
            ['SD1a', 'SD1b', 'SD1c'], station=station, minstrument='keithley1.amplitude')
        _ = sensing_dot.autoTune(step=-8, fig=None)
        qtt.simulation.virtual_dot_array.close()

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
