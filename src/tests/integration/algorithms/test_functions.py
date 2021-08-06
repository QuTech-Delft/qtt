import unittest

import matplotlib.pyplot as plt
import numpy as np

import qtt
from qtt.algorithms.functions import (Fermi,
                                      _estimate_exp_decay_initial_parameters,
                                      cost_exp_decay, double_gaussian,
                                      estimate_parameters_damped_sine_wave,
                                      exp_function, fit_exp_decay,
                                      fit_gauss_ramsey, gauss_ramsey, gaussian,
                                      linear_function, logistic,
                                      plot_gauss_ramsey_fit)


class TestFunctions(unittest.TestCase):

    def test_estimate_dominant_frequency(self, fig=1):
        y_data = np.array([0.122, 0.2, 0.308, 0.474, 0.534, 0.618, 0.564, 0.436, 0.318,
                           0.158, 0.13, 0.158, 0.336, 0.434, 0.51, 0.59, 0.592, 0.418,
                           0.286, 0.164, 0.156, 0.186, 0.25, 0.362, 0.524])
        sample_rate = 47368421
        estimated_frequency = qtt.algorithms.functions.estimate_dominant_frequency(
            y_data, sample_rate=sample_rate, fig=fig)
        self.assertAlmostEqual(estimated_frequency, 4750425, places=-2)
        plt.close(fig=fig)

    def test_fit_gauss_ramsey(self, fig=100):
        y_data = np.array([0.6019, 0.5242, 0.3619, 0.1888, 0.1969, 0.3461, 0.5276, 0.5361,
                           0.4261, 0.28, 0.2323, 0.2992, 0.4373, 0.4803, 0.4438, 0.3392,
                           0.3061, 0.3161, 0.3976, 0.4246, 0.398, 0.3757, 0.3615, 0.3723,
                           0.3803, 0.3873, 0.3873, 0.3561, 0.37, 0.3819, 0.3834, 0.3838,
                           0.37, 0.383, 0.3573, 0.3869, 0.3838, 0.3792, 0.3757, 0.3815])
        x_data = np.array([i * 1.6 / 40 for i in range(40)])

        par_fit_test, _ = fit_gauss_ramsey(x_data * 1e-6, y_data)

        plot_gauss_ramsey_fit(x_data*1e-6, y_data, par_fit_test, fig=fig)

    def test_fit_gauss_ramsey_weights(self, fig=123):
        x_data = np.hstack((np.linspace(0, 3 * np.pi, 20), np.linspace(3 * np.pi, 4 * np.pi, 100)))
        y_data = np.sin(x_data)
        y_data[21:] = 0

        fitted_parameters_unweighted, _ = fit_gauss_ramsey(x_data, y_data)
        fitted_parameters, _ = fit_gauss_ramsey(x_data, y_data, weight_power=10)
        self.assertAlmostEqual(fitted_parameters[0], 1, places=2)
        self.assertTrue(fitted_parameters[1] > 1e3)
        self.assertAlmostEqual(fitted_parameters[2], 1 / (2 * np.pi), places=1)
        self.assertAlmostEqual(fitted_parameters[3], 0, places=6)
        self.assertAlmostEqual(fitted_parameters[4], 0, places=6)
        if fig is not None:
            plt.figure(fig)
            plt.clf()
            plt.plot(x_data, y_data, '.')
            plt.plot(x_data, gauss_ramsey(x_data, fitted_parameters_unweighted), 'b')
            plt.plot(x_data, gauss_ramsey(x_data, fitted_parameters), 'm')
            plt.close(fig)
