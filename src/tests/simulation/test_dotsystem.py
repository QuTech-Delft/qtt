""" Test simulation of a coupled dot system

"""

# %% Load packages

import numpy as np
import unittest
from qtt.simulation.dotsystem import OneDot, DoubleDot,  TripleDot, FourDot, TwoXTwo


class TestDotSystem(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_basic_dots(self):
        dot_system = OneDot()
        self.assertEqual(dot_system.ndots, 1)
        self.assertEqual(dot_system.varnames, ['det1', 'onsiteC1'])
        dot_system = DoubleDot()
        self.assertEqual(dot_system.varnames, ['det1', 'det2', 'onsiteC1', 'onsiteC2', 'intersiteC12', 'tun1'])
        self.assertEqual(dot_system.ndots, 2)
        dot_system = TripleDot(maxelectrons=2)
        self.assertEqual(dot_system.ndots, 3)
        dot_system = FourDot(maxelectrons=2)
        self.assertEqual(dot_system.ndots, 4)
        dot_system = TwoXTwo()

    def test_triple_dot_basics(self):
        dot_system = TripleDot(maxelectrons=2)

        paramvalues_2d = np.zeros((3, 4, 5))
        _ = dot_system.simulate_honeycomb(paramvalues_2d, multiprocess=False)
        self.assertEqual(dot_system.honeycomb.shape, paramvalues_2d.shape[1:])

    def test_twoxtwo(self):
        np.random.seed(2019)
        m = TwoXTwo()
        energies = m.calculate_energies(list(np.random.rand(m.ndots)))
        self.assertEqual(energies.size, 81)
        self.assertAlmostEqual(energies.min(), -5.116820166778993)
        self.assertAlmostEqual(energies.max(), 0.0)
        energies, eigenstates = m.solveH()
        self.assertEqual(energies.size, 81)
        self.assertEqual(eigenstates.size, 6561)
        self.assertAlmostEqual(eigenstates.min(), 0.0)
        self.assertAlmostEqual(eigenstates.max(), 1.0)
        # m.showstates(81)
