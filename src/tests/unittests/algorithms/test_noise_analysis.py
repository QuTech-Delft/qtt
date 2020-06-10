import unittest

import numpy as np

from qtt.algorithms.noise_analysis import (power_law_model, outlier_detection, calculate_psd_welch, plot_psd,
    generate_pink_noise, fit_power_law, fit_power_law_loglog)


class TestNoise(unittest.TestCase):

    def setUp(self):
        sample_rate = 10e3
        N = int(1e5)
        freq = 1234.0

        self.pink_noise = generate_pink_noise(sample_rate, N, A=2.5, alpha=1.2) + 2 * \
            np.sin(2 * np.pi * freq * np.arange(N) / sample_rate)
        self.sample_rate = sample_rate

    def test_power_law_model(self):
        A = 2
        alpha = 1.1
        frequencies = np.arange(0.1, 100, 7)
        values = power_law_model(frequencies, A, alpha)
        self.assertIsInstance(values, np.ndarray)
        np.testing.assert_array_almost_equal(values, A / frequencies**alpha)

    def test_outlier_detection(self):
        data=np.random.rand(16)
        data[2]=10
        data[14]=-4
        inliers = outlier_detection(data)
        self.assertSequenceEqual(list(np.logical_not(inliers).nonzero()[0]), [2, 14])

    def test_outlier_detection_threshold(self):
        data=[0,1,-.2,-2]
        inliers = outlier_detection(data, threshold=.5)
        self.assertListEqual(list(inliers), [ True, False,  True, False])

    def test_outlier_detection_empty_data(self):
        data=[]
        with self.assertRaises(IndexError):
            _ = outlier_detection(data)

    def test_outlier_detection_empty_data_threshold(self):
        data=[]
        inliers = outlier_detection(data, threshold=1)
        self.assertListEqual(list(inliers), [])


    def test_calculate_psd_welch(self):
        f_welch, psd_welch = calculate_psd_welch(self.pink_noise, sample_rate=self.sample_rate, nperseg=512)
        plot_psd(f_welch, psd_welch, fig=1)
        self.assertEqual(f_welch[0], 0)
        self.assertEqual(f_welch[-1], 5000.)

    def test_fit_power_law(self):
        f_welch, psd_welch = calculate_psd_welch(self.pink_noise, sample_rate=self.sample_rate, nperseg=512)
        fitted_parameters, results = fit_power_law(f_welch[1:], psd_welch[1:])
        self.assertIsInstance(results, dict)
        np.testing.assert_array_equal(fitted_parameters, results['fitted_parameters'])
        self.assertIsNone(results['inliers'])

        fitted_parameters, results = fit_power_law(f_welch[1:], psd_welch[1:], remove_outliers=True)
        self.assertIsInstance(results, dict)
        np.testing.assert_array_equal(fitted_parameters, results['fitted_parameters'])
        self.assertAlmostEqual(fitted_parameters[1], 1.2, places=1)

        with self.assertRaises(Exception):
            fit_pink_noise(f_welch, psd_welch)

    def test_fit_power_law_loglog(self):
        f_welch, psd_welch = calculate_psd_welch(self.pink_noise, sample_rate=self.sample_rate, nperseg=512)
        fitted_parameters, results = fit_power_law_loglog(f_welch[1:], psd_welch[1:])
        self.assertIsInstance(results, dict)
        np.testing.assert_array_equal(fitted_parameters, results['fitted_parameters'])
        self.assertIsNone(results['inliers'])

        fitted_parameters, results = fit_power_law_loglog(f_welch[1:], psd_welch[1:], remove_outliers=True)
        self.assertIsInstance(results, dict)
        np.testing.assert_array_equal(fitted_parameters, results['fitted_parameters'])
        self.assertAlmostEqual(fitted_parameters[1], 1.2, places=1)

        with self.assertRaises(Exception):
            fit_pink_noise(f_welch, psd_welch)

