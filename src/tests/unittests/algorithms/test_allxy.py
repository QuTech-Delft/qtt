import itertools
import numpy as np
import unittest
import matplotlib.pyplot as plt

from qtt.data import makeDataSet1Dplain
from qtt.algorithms.allxy import fit_allxy, plot_allxy, allxy_model


class TestAllxy(unittest.TestCase):

    def test_allxy(self, fig=None):
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

    def test_allxy_with_init_params(self, fig=None):
        dataset = makeDataSet1Dplain('index', np.arange(21), 'allxy', [0.12, 0.16533333, 0.136, 0.17066666, 0.20266667,
                                                                       0.452, 0.48133334, 0.58666666, 0.43199999, 0.52933334,
                                                                       0.44533333, 0.51066667, 0.46, 0.48133334, 0.47066667,
                                                                       0.47333333, 0.488, 0.80799999, 0.78933333, 0.788,
                                                                       0.79333333])
        init_params = np.array([0.15893333, 0., 0.48422222, 0., 0.79466666, 0.])
        result = fit_allxy(dataset, init_params)
        self.assertIsInstance(result, dict)
        np.testing.assert_array_almost_equal([0.1, 0, .5, 0, .9, 0], result['fitted_parameters'], decimal=1)
        if fig is not None:
            plot_allxy(dataset, result, fig=fig, plot_initial_estimate=True)
            plt.close(fig)

    def test_allxy_covariance_regression(self):
        allxy_data = [0.175, 0.24166666666666667, 0.23166666666666666, 0.2, 0.21, 0.49666666666666665,
                      0.5316666666666666, 0.545, 0.53, 0.605, 0.5466666666666666, 0.58, 0.505, 0.5916666666666667, 0.6,
                      0.53, 0.5716666666666667, 0.9016666666666666, 0.91, 0.905, 0.9033333333333333]
        dataset = makeDataSet1Dplain('index', np.arange(21), 'allxy', allxy_data)
        result = fit_allxy(dataset)
        self.assertTrue(result['fitted_parameters_covariance'] is not None)
        self.assertAlmostEqual(result['reduced_chi_squared'], 0.000976, places=5)
        self.assertAlmostEqual(result['fitted_parameters_covariance'][0], 1.95305266e-04, places=5)

    def test_allxy_model(self):
        offsets = [.0, .1, .2]
        slopes = [-.1, 0, .1]

        indices = range(5)
        for offset, slope, idx in itertools.product(offsets, slopes, indices):
            mean_index = np.mean(indices)
            self.assertAlmostEqual(allxy_model(idx, offset, slope, 1, 2, 3, 4),
                                   offset + slope * (idx - mean_index))

        indices = range(5, 17)
        for offset, slope, idx in itertools.product(offsets, slopes, indices):
            mean_index = np.mean(indices)
            self.assertAlmostEqual(allxy_model(idx, -1, -2, offset, slope, 3, 4),
                                   offset + slope * (idx - mean_index))

        indices = range(17, 2)
        for offset, slope, idx in itertools.product(offsets, slopes, indices):
            mean_index = np.mean(indices)
            self.assertAlmostEqual(allxy_model(idx, -1, -2, -3, -4, offset, slope),
                                   offset + slope * (idx - mean_index))
