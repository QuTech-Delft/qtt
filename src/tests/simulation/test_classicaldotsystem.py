# -*- coding: utf-8 -*-
""" Test classical Quantum Dot Simulator

"""

import unittest
import numpy as np

from qtt.simulation.classicaldotsystem import MultiDot


class TestClassicalDotSystem(unittest.TestCase):

    def test_classical_dot_system(self):
        np.random.seed(2019)
        m = MultiDot('multidot', 4, maxelectrons=3)
        energies = m.calculate_energies(np.random.rand(m.ndots))

        self.assertEqual(energies.size, 256)
        self.assertAlmostEqual(energies.min(), 0.0)
        self.assertAlmostEqual(energies.max(), 1672.026261148515, 6)
        self.assertAlmostEqual(energies.sum(), 153915.72102190088, 3)
        m.solve()
        # m.showstates(8)
