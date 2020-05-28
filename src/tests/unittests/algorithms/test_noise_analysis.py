import io
import unittest
from unittest import mock
from unittest.mock import call, patch

import numpy as np

from qtt.algorithms.noise_analysis import calculate_psd_welch, fit_pink_noise
from qtt.algorithms.noise_analysis import pink_noise_model, outlier_detection, calculate_psd_welch, plot_psd, generate_pink_noise, fit_pink_noise


class TestNoise(unittest.TestCase):

    def setUp(self):
        sample_rate = 10e3
        N = int(1e5)
        freq = 1234.0

        self.pink_noise = generate_pink_noise(sample_rate, N, A=2.5, alpha=1.2) + 2 * \
            np.sin(2 * np.pi * freq * np.arange(N) / sample_rate)
        self.sample_rate = sample_rate

    def test_pink_noise_model(self):
        A = 2
        alpha = 1.1
        frequencies = np.arange(0.1, 100, 7)
        values = pink_noise_model(frequencies, A, alpha)
        self.assertIsInstance(values, np.ndarray)
        np.testing.assert_array_almost_equal(values, A / frequencies**alpha)

    def test_calculate_psd_welch(self):
        f_welch, psd_welch = calculate_psd_welch(self.pink_noise, sample_rate=self.sample_rate, nperseg=512)
        plot_psd(f_welch, psd_welch, fig=1)
        self.assertEqual(f_welch[0], 0)
        self.assertEqual(f_welch[-1], 5000.)

    def test_fit_pink_noise(self):
        f_welch, psd_welch = calculate_psd_welch(self.pink_noise, sample_rate=self.sample_rate, nperseg=512)
        fitted_parameters, results = fit_pink_noise(f_welch[1:], psd_welch[1:])
        self.assertIsInstance(results, dict)
        np.testing.assert_array_equal(fitted_parameters, results['fitted_parameters'])
        self.assertIsNone(results['inliers'])

        fitted_parameters, results = fit_pink_noise(f_welch[1:], psd_welch[1:], remove_outliers=True)
        self.assertIsInstance(results, dict)
        np.testing.assert_array_equal(fitted_parameters, results['fitted_parameters'])
        self.assertAlmostEqual(fitted_parameters[1], 1.2, places=1)

        with self.assertRaises(Exception):
            fit_pink_noise(f_welch, psd_welch)
