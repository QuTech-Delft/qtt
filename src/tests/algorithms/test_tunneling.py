""" Test functionality for analysing inter-dot tunnel frequencies

"""

from unittest import TestCase
import numpy as np
from qtt.algorithms.tunneling import polmod_all_2slopes, fit_pol_all


class TestTunneling(TestCase):

    def test_polFitting(self):
        """ Test the polarization fitting. """
        np.random.seed(2019)
        x_data = np.linspace(-100, 100, 1000)
        k_t = 6.5
        par_init = np.array([20, 2, 100, -.5, -.45, 300])
        y_data = polmod_all_2slopes(x_data, par_init, k_t)
        noise = np.random.normal(0, 3, y_data.shape)
        par_fit, _, _ = fit_pol_all(x_data, y_data + noise, k_t, par_guess=par_init)
        self.assertEqual(par_fit.size, 6)
        self.assertAlmostEqual(par_fit.min(), -0.502742506913356)
        self.assertAlmostEqual(par_fit.max(), 303.2601434943597)
        self.assertAlmostEqual(par_fit.sum(), 423.8971145696139)
        np.testing.assert_allclose(par_fit[0], par_init[0], .1)
