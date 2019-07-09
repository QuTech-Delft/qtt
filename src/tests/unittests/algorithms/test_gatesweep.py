""" Test functionality to analyse pinch-off scans."""

# %% Load packages

import unittest
import numpy as np
import matplotlib.pyplot as plt
from qtt.data import makeDataSet1Dplain
from qtt.algorithms.functions import logistic
from qtt.algorithms.gatesweep import analyseGateSweep, plot_pinchoff


class TestGateSweep(unittest.TestCase):

    def test_analyseGateSweep(self, fig=None, verbose=0):
        x = np.arange(-800, 0, 1)  # mV
        y = logistic(x, x0=-400, alpha=.05)
        data_set = makeDataSet1Dplain('plunger', x, 'current', y)
        result = analyseGateSweep(data_set, fig=fig, verbose=verbose)
        self.assertAlmostEqual(result['_pinchvalueX'], -450.0)
        self.assertAlmostEqual(result['lowvalue'], 9.445888759986548e-18, 6)
        self.assertAlmostEqual(result['highvalue'], 0.9999999999999998, 6)
        self.assertAlmostEqual(result['midvalue'], 0.29999999999999993, 6)
        self.assertEqual(result['type'], 'gatesweep')
        if fig:
            plot_pinchoff(result, ds=data_set, fig=fig)
        plt.close('all')
