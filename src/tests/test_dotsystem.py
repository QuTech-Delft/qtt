import unittest
import numpy as np

import qtt.data
import qtt.measurements.scans
from qtt.simulation.dotsystem import TripleDot, FourDot, TwoXTwo


class TestDotSystem(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_basic_dots(self):
        dotsystem = qtt.simulation.dotsystem.OneDot()
        self.assertEqual(dotsystem.ndots, 1)
        self.assertEqual(dotsystem.varnames, ['det1', 'onsiteC1'])
        dotsystem = qtt.simulation.dotsystem.DoubleDot()
        self.assertEqual(dotsystem.varnames, ['det1', 'det2', 'onsiteC1', 'onsiteC2', 'intersiteC12', 'tun1'])
        self.assertEqual(dotsystem.ndots, 2)
        dotsystem = TripleDot(maxelectrons=2)
        self.assertEqual(dotsystem.ndots, 3)
        dotsystem = FourDot(maxelectrons=2)
        self.assertEqual(dotsystem.ndots, 4)
        dotsystem = TwoXTwo()

    def test_triple_dot_basics(self):
        dotsystem = TripleDot(maxelectrons=2)

        paramvalues2D = np.zeros((3, 4, 5))
        _ = dotsystem.simulate_honeycomb(paramvalues2D, multiprocess=False)
        self.assertEqual(dotsystem.honeycomb.shape, paramvalues2D.shape[1:])


if __name__ == '__main__':

    unittest.main()
    dotsystem = TripleDot(maxelectrons=2)
    self = dotsystem
