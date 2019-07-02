""" Mathematical functions and models."""

import unittest
import matplotlib.pyplot as plt
import numpy as np
from qtt.algorithms.functions import fit_gauss_ramsey, gauss_ramsey


class TestFunctions(unittest.TestCase):

    def test_fit_gauss_ramsey(self, fig=100):
        y_data = np.array([0.6019, 0.5242, 0.3619, 0.1888, 0.1969, 0.3461, 0.5276, 0.5361,
                           0.4261, 0.28, 0.2323, 0.2992, 0.4373, 0.4803, 0.4438, 0.3392,
                           0.3061, 0.3161, 0.3976, 0.4246, 0.398, 0.3757, 0.3615, 0.3723,
                           0.3803, 0.3873, 0.3873, 0.3561, 0.37, 0.3819, 0.3834, 0.3838,
                           0.37, 0.383, 0.3573, 0.3869, 0.3838, 0.3792, 0.3757, 0.3815])
        x_data = np.array([i * 1.6 / 40 for i in range(40)])

        par_fit_test, _ = fit_gauss_ramsey(x_data * 1e-6, y_data)

        self.assertTrue(np.abs(np.abs(par_fit_test[0]) - 0.21) < 0.1)
        self.assertTrue(np.abs(par_fit_test[-2] - 1.88) < 0.1)
        self.assertTrue(np.abs(par_fit_test[-1] - 0.38) < 0.1)

        test_x = np.linspace(0, x_data.max() * 1e-6, 200)
        test_y = gauss_ramsey(test_x, par_fit_test)

        if fig is not None:
            plt.figure(10)
            plt.clf()
            plt.plot(x_data, y_data, 'o', label='input data')
            plt.plot(test_x * 1e6, test_y, label='fitted curve')
            plt.legend(numpoints=1)
            plt.close('all')
