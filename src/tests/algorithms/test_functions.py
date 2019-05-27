""" Mathematical functions and models."""

import unittest
import matplotlib.pyplot as plt
import numpy as np
from qtt.algorithms.functions import gaussian, fit_gaussian, fit_double_gaussian, double_gaussian, exp_function, \
    fit_gauss_ramsey, gauss_ramsey, cost_exp_decay, logistic, linear_function, Fermi, fit_exp_decay


class TestFunctions(unittest.TestCase):

    @staticmethod
    def test_fit_exp_decay():
        x_data = np.arange(0, 10., .1)
        parameters = [0, 1, 1]
        y_data = exp_function(x_data, *parameters)
        fitted = fit_exp_decay(x_data, y_data)
        np.testing.assert_array_almost_equal(fitted, parameters, decimal=3)
        fitted = fit_exp_decay(x_data, y_data, offset_parameter=0.1)
        np.testing.assert_array_almost_equal(fitted, [.1, .95, 1.4], decimal=1)

    @staticmethod
    def test_logistic():
        y = logistic(0.0, 1.0, alpha=1.0)
        np.testing.assert_almost_equal(y, 0.11920292202211755, decimal=6)

    def test_Fermi(self):
        values = Fermi(np.array([0, 1, 2]), 0, 1, 2, kb=1)
        np.testing.assert_array_almost_equal(values, np.array([0.5, 0.37754067, 0.26894142]))

        value = Fermi(10., 0, 1, 2, kb=10)
        self.assertAlmostEqual(value, 0.3775406687981454, 6)

    def test_fit_gaussian(self):
        x_data = np.linspace(0, 10, 100)
        gauss_data = gaussian(x_data, mean=4, std=1, amplitude=5)
        noise = np.random.rand(100)
        [mean, s, amplitude, offset], _ = fit_gaussian(x_data=x_data, y_data=(gauss_data + noise))
        self.assertTrue(3.5 < mean < 4.5)
        self.assertTrue(0.5 < s < 1.5)
        self.assertTrue(4.5 < amplitude < 5.5)
        self.assertTrue(0.0 < offset < 1.0)

    def test_fit_double_gaussian(self):
        x_data = np.arange(-4, 4, .05)
        initial_parameters = [10, 20, 1, 1, -2, 2]
        y_data = double_gaussian(x_data, initial_parameters)

        fitted_parameters, _ = fit_double_gaussian(x_data, y_data)
        parameter_diff = np.abs(fitted_parameters - initial_parameters)
        self.assertTrue(np.all(parameter_diff < 1e-3))

    def test_cost_exp_decay(self):
        params = [0, 1, 1]
        x_data = np.arange(0, 20)
        y_data = exp_function(x_data, *params)
        y_data[-1] += 10
        c = cost_exp_decay(x_data, y_data, params)
        self.assertTrue(c == 10.0)
        c = cost_exp_decay(x_data, y_data, params, threshold=None)
        self.assertTrue(c == 10.0)
        c = cost_exp_decay(x_data, y_data, params, threshold='auto')
        self.assertTrue(c < 10.0)

    def test_fit_gauss_ramsey(self, fig=None):
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

    def test_logistic_and_linear_function(self):
        x_data = np.arange(-10, 10, 0.1)

        _ = logistic(x_data, x0=0, alpha=1)
        self.assertTrue(logistic(0, x0=0, alpha=1) == 0.5)

        _ = linear_function(x_data, 1, 2)
        self.assertTrue(linear_function(0, 1, 2) == 2)
        self.assertTrue(linear_function(3, 1, 2) == 5)
