""" Test functions to analyse anti-crossings in charge stability diagrams

"""

# %%
import unittest
import numpy as np
from qtt.algorithms.anticrossing import fit_anticrossing
from qtt.data import makeDataSet2Dplain
# %%


class TestAntiCrossing(unittest.TestCase):

    def test_anti_crossing(self):
        np.random.seed(2019)
        nx = 30
        ny = 40
        dsx = makeDataSet2Dplain('x', .5 * np.arange(nx),
                                 'y', .5 * np.arange(ny),
                                 'z', np.random.rand(ny, nx,))
        fitdata = fit_anticrossing(dsx, verbose=0)
        self.assertEqual(len(fitdata), 5)
        self.assertEqual(fitdata['labels'][0], 'y')
        self.assertEqual(fitdata['labels'][1], 'x')
        self.assertAlmostEqual(float(fitdata['fitpoints']['centre'][0]), 9.48638335851373, 2)
        self.assertAlmostEqual(float(fitdata['fitpoints']['centre'][1]), 12.168880945142195, 2)
        self.assertAlmostEqual(float(fitdata['fitpoints']['left_point'][0]), 7.148586295647548, 2)
        self.assertAlmostEqual(float(fitdata['fitpoints']['left_point'][1]), 9.831083882276012, 2)
        self.assertAlmostEqual(float(fitdata['fitpoints']['right_point'][0]), 11.824180421379914, 2)
        self.assertAlmostEqual(float(fitdata['fitpoints']['right_point'][1]), 14.506678008008379, 2)
        self.assertAlmostEqual(float(fitdata['fit_params'].min()), 0.477501117732345, 2)
        self.assertAlmostEqual(float(fitdata['fit_params'].max()), 9.61138335851373, 2)
