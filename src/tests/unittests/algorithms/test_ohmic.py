""" Functionality to test fit scans of ohmic contacts."""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from qtt.algorithms.ohmic import fitOhmic
from qtt.data import makeDataSet1Dplain


class TestOhmic(unittest.TestCase):

    def test_fitohmic(self, fig=None):
        np.random.seed(2019)
        x = np.arange(-200, 200)
        y = 1e-10 * (x + 50 + 20 * np.random.rand(x.size))
        ds = makeDataSet1Dplain('gate', x, xunit='mV', yname='current', y=y)
        r = fitOhmic(ds, fig=fig, gainx=1e-6, gainy=1)
        self.assertAlmostEqual(r['resistance'], 10006.210459347802, 6)
        self.assertAlmostEqual(r['biascurrent'], 5.997659339645733e-09, 6)
        self.assertEqual(r['description'], 'ohmic')
        self.assertEqual(r['fitparam'][0], 9.993793395238855e-05, 6)
        self.assertEqual(r['fitparam'][1], 5.997659339645733e-09, 6)
        plt.close('all')