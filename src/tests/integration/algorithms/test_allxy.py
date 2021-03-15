import itertools
import numpy as np
import unittest
import matplotlib.pyplot as plt

from qtt.data import makeDataSet1Dplain
from qtt.algorithms.allxy import fit_allxy, plot_allxy, allxy_model


class TestAllxy(unittest.TestCase):

    def test_allxy(self, fig=1):
        dataset = makeDataSet1Dplain('index', np.arange(21), 'allxy', [0.12, 0.16533333, 0.136, 0.17066666, 0.20266667,
                                                                       0.452, 0.48133334, 0.58666666, 0.43199999,
                                                                       0.52933334, 0.44533333, 0.51066667, 0.46,
                                                                       0.48133334, 0.47066667, 0.47333333, 0.488,
                                                                       0.80799999, 0.78933333, 0.788, 0.79333333])
        result = fit_allxy(dataset)
        self.assertIsInstance(result, dict)
        np.testing.assert_array_almost_equal([0.1, 0, .5, 0, .9, 0], result['fitted_parameters'], decimal=1)
        if fig is not None:
            plot_allxy(dataset, result, fig=fig, plot_initial_estimate=True)
            plt.close(fig)
