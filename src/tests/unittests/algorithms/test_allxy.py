import numpy as np
import unittest
import matplotlib.pyplot as plt

import qtt
from qtt.algorithms.allxy import fit_allxy, plot_allxy


class TestAllxy(unittest.TestCase):

    def test_allxy(self):
        dataset = qtt.data.makeDataSet1Dplain('index', np.arange(21), 'allxy', [0.12, 0.16533333, 0.136, 0.17066666, 0.20266667,
                                                                                0.452, 0.48133334, 0.58666666, 0.43199999, 0.52933334,
                                                                                0.44533333, 0.51066667, 0.46, 0.48133334, 0.47066667,
                                                                                0.47333333, 0.488, 0.80799999, 0.78933333, 0.788,
                                                                                0.79333333])
        result = fit_allxy(dataset)
        plot_allxy(dataset, result, fig=1)
        plt.close(1)
        self.assertIsInstance(result, dict)
        self.assertTrue(result['fitted_parameters'][0] < 0.2)
