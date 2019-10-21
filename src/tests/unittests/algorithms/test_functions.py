import unittest

import numpy as np
from qtt.algorithms.functions import gaussian, fit_gaussian, fit_double_gaussian, double_gaussian, exp_function, \
    fit_gauss_ramsey, cost_exp_decay, logistic, linear_function, Fermi, fit_exp_decay, \
    _estimate_exp_decay_initial_parameters, plot_gauss_ramsey_fit, estimate_parameters_gauss_ramsey, \
    estimate_dominant_frequency


class TestData:
    class GaussRamsey:
        @staticmethod
        def negative_frequency_1():
            return {
                'x_data': np.array([0., 0.0225, 0.045, 0.06708334, 0.08958333,
                                    0.11208333, 0.13458334, 0.15708333, 0.17916666, 0.20166667,
                                    0.22416666, 0.24666667, 0.26916668, 0.29124999, 0.31375,
                                    0.33625001, 0.35874999, 0.38083333, 0.40333334, 0.42583334,
                                    0.44833332, 0.47083333, 0.49291667, 0.51541668, 0.53791666,
                                    0.56041664, 0.58291668, 0.60500002, 0.6275, 0.64999998]) * 1E-6,

                'y_data': np.array([0.12, 0.14666666, 0.24666667, 0.28666666, 0.36333334,
                                    0.41166666, 0.41, 0.37, 0.28666666, 0.19,
                                    0.18333334, 0.14666666, 0.21666667, 0.29833335, 0.37666667,
                                    0.41999999, 0.43666667, 0.43833333, 0.33333334, 0.285,
                                    0.24666667, 0.16833334, 0.19833334, 0.20166667, 0.28166667,
                                    0.37, 0.41999999, 0.45500001, 0.38833332, 0.39333335])
            }

        @staticmethod
        def negative_frequency_2():
            return {
                'x_data': np.array([0., 0.0225, 0.045, 0.06708334, 0.08958333,
                                    0.11208333, 0.13458334, 0.15708333, 0.17916666, 0.20166667,
                                    0.22416666, 0.24666667, 0.26916668, 0.29124999, 0.31375,
                                    0.33625001, 0.35874999, 0.38083333, 0.40333334, 0.42583334,
                                    0.44833332, 0.47083333, 0.49291667, 0.51541668, 0.53791666,
                                    0.56041664, 0.58291668, 0.60500002, 0.6275, 0.64999998]) * 1E-6,

                'y_data': np.array([0.09833334, 0.13500001, 0.19333333, 0.27166668, 0.37833333,
                                    0.405, 0.39166668, 0.33833334, 0.23333333, 0.16833334,
                                    0.11166666, 0.11, 0.20166667, 0.22166666, 0.29166666,
                                    .35833332, 0.42666668, 0.39500001, 0.32833335, 0.27833334,
                                    0.17166667, 0.12166667, 0.12333333, 0.19333333, 0.21333334,
                                    0.33833334, 0.42833334, 0.40833333, 0.41, 0.34833333])
            }

        @staticmethod
        def negative_frequency_3():
            return {
                'x_data': np.array([0.00000000e+00, 2.60000000e-07, 5.20000000e-07, 7.80000000e-07,
                                    1.04666667e-06, 1.30666667e-06, 1.56666667e-06, 1.82666667e-06,
                                    2.08666667e-06, 2.34666667e-06, 2.60666667e-06, 2.86666667e-06,
                                    3.13333333e-06, 3.39333333e-06, 3.65333333e-06, 3.91333333e-06,
                                    4.17333333e-06, 4.43333333e-06, 4.69333333e-06, 4.95333333e-06,
                                    5.22000000e-06, 5.48000000e-06, 5.74000000e-06, 6.00000000e-06]),

                'y_data': np.array([0.475, 0.13, 0.42, 0.24833333, 0.20833333,
                                    0.40833333, 0.12666667, 0.37833333, 0.29, 0.23,
                                    0.38166667, 0.17333333, 0.37166667, 0.28333333, 0.25166667,
                                    0.33166667, 0.22666667, 0.31666667, 0.27833333, 0.285,
                                    0.29166667, 0.22833333, 0.33, 0.28833333])
            }

        @staticmethod
        def fit_high_freq_low_amp_1():
            return {
                'x_data': np.array([0., 0.03, 0.06, 0.09, 0.12,
                                    0.15000001, 0.18000001, 0.20999999, 0.23999999, 0.27000001,
                                    0.30000001, 0.33000001, 0.36000001, 0.38999999, 0.41999999,
                                    0.44999999, 0.47999999, 0.50999999, 0.54000002, 0.56999999,
                                    0.60000002]) * 1E-6,

                'y_data': np.array([0.175, 0.22833334, 0.435, 0.5933333, 0.73833334,
                                    0.65833336, 0.47499999, 0.315, 0.14166667, 0.19,
                                    0.39833334, 0.53833336, 0.71499997, 0.685, 0.59500003,
                                    0.375, 0.25166667, 0.19499999, 0.28833333, 0.47333333,
                                    0.63166666])
            }
    # end of GaussRamsey


# end of TestData


class TestFunctions(unittest.TestCase):

    def test_estimate_dominant_frequency(self):
        y_data = np.array([0.122, 0.2, 0.308, 0.474, 0.534, 0.618, 0.564, 0.436, 0.318,
                           0.158, 0.13, 0.158, 0.336, 0.434, 0.51, 0.59, 0.592, 0.418,
                           0.286, 0.164, 0.156, 0.186, 0.25, 0.362, 0.524])
        sample_rate = 47368421
        estimated_frequency = estimate_dominant_frequency(y_data, sample_rate=sample_rate)
        self.assertAlmostEqual(estimated_frequency, 5684210, places=-1)

    @staticmethod
    def test_fit_exp_decay():
        parameters = [0, 1, 1]
        x_data = np.arange(0, 10., .1)
        y_data = exp_function(x_data, *parameters)
        fitted = fit_exp_decay(x_data, y_data)
        np.testing.assert_array_almost_equal(fitted, parameters, decimal=3)
        fitted = fit_exp_decay(x_data, y_data, offset_parameter=0.1)
        np.testing.assert_array_almost_equal(fitted, [.1, .95, 1.4], decimal=1)

    @staticmethod
    def test_fit_exp_decay_upward():
        parameters = [0, -1, 1]
        x_data = np.arange(0, 10., .1)
        y_data = exp_function(x_data, *parameters)
        fitted = fit_exp_decay(x_data, y_data)
        np.testing.assert_array_almost_equal(fitted, np.array(
            [5.38675880e-05, -9.99998574e-01, 9.99760970e-01]), decimal=3)

    @staticmethod
    def test_fit_exp_decay_shifted_xdata():
        x_data = np.array([0.0e+00, 1.0e-05, 2.0e-05, 3.0e-05, 4.0e-05, 5.0e-05, 6.0e-05,
                           7.0e-05, 8.0e-05, 9.0e-05, 1.0e-04, 1.1e-04, 1.2e-04, 1.3e-04,
                           1.4e-04, 1.5e-04, 1.6e-04, 1.7e-04, 1.8e-04, 1.9e-04, 2.0e-04,
                           2.1e-04, 2.2e-04, 2.3e-04, 2.4e-04])
        y_data = np.array([-1.01326172, -0.80266628, -0.68311867, -0.55951015, -0.43239706,
                           -0.37280919, -0.30871727, -0.25232649, -0.21540011, -0.1532993,
                           -0.15151554, -0.09949812, -0.07043394, -0.07213127, -0.07627919,
                           -0.0440114, -0.05251701, -0.06129484, -0.04697476, -0.03194123,
                           -0.02915847, -0.02098205, -0.01857533, 0.00326683, -0.0185229])
        initial_parameters0 = _estimate_exp_decay_initial_parameters(x_data, y_data, None)
        initial_parameters = _estimate_exp_decay_initial_parameters(x_data - .1, y_data, None)
        np.testing.assert_array_almost_equal(initial_parameters0, np.array(
            [-7.19224040e-03, -9.04983668e-01, 8.33333333e+03]), 3)
        np.testing.assert_array_almost_equal(initial_parameters, np.array(
            [-7.19224040e-03, -0.00000000e+00, 8.33333333e+03]), 3)

    @staticmethod
    def test_logistic():
        y = logistic(0.0, 1.0, alpha=1.0)
        np.testing.assert_almost_equal(y, 0.11920292202211755, decimal=6)

    def test_Fermi(self):
        values = Fermi(np.array([0, 1, 2]), 0, 1, 2, kb=1)
        np.testing.assert_array_almost_equal(values, np.array([0.5, 0.37754067, 0.26894142]))

        value = Fermi(10., 0, 1, 2, kb=10)
        self.assertAlmostEqual(value, 0.3775406687981454, 6)

    def test_estimate_parameters_gauss_ramsey(self):
        y_data = np.array([0.25285714, 0.31714286, 0.48857143, 0.66285714, 0.77857143,
                           0.72857143, 0.58714286, 0.42571429, 0.28142857, 0.29571429,
                           0.39428571, 0.47285714, 0.56857143, 0.70428571, 0.76857143,
                           0.73571429, 0.62714286, 0.49714286, 0.42857143, 0.30714286,
                           0.31, 0.31714286, 0.37285714])

        x_data = np.array([0.00000000e+00, 6.81818182e-09, 1.36363636e-08, 2.04545455e-08,
                           2.72727273e-08, 3.40909091e-08, 4.09090909e-08, 4.77272727e-08,
                           5.45454545e-08, 6.13636364e-08, 6.81818182e-08, 7.50000000e-08,
                           8.18181818e-08, 8.86363636e-08, 9.54545455e-08, 1.02272727e-07,
                           1.09090909e-07, 1.15909091e-07, 1.22727273e-07, 1.29545455e-07,
                           1.36363636e-07, 1.43181818e-07, 1.50000000e-07])

        estimated_parameters = estimate_parameters_gauss_ramsey(x_data, y_data)
        self.assertFalse(np.any(np.isnan(estimated_parameters)))
        self.assertAlmostEqual(estimated_parameters[0], 0.263, places=1)

    def test_estimate_parameters_damped_sine_wave_degenerate(self):
        x_data = np.array([0., 1., 2.])
        y_data = np.array([0, 0, 0])

        estimated_parameters = estimate_parameters_gauss_ramsey(x_data, y_data)
        self.assertEqual(estimated_parameters[0], 0.)

    def test_estimate_parameters_gauss_ramsey_exact(self):
        x_data = np.arange(0, 20., .12)
        exact_frequency = .4
        exact_offset = 2.2
        y_data = np.sin(2 * np.pi * exact_frequency * x_data) + exact_offset

        estimated_parameters = estimate_parameters_gauss_ramsey(x_data, y_data)
        self.assertAlmostEqual(estimated_parameters[0], 1., places=1)
        self.assertAlmostEqual(estimated_parameters[2], exact_frequency, places=1)
        self.assertAlmostEqual(estimated_parameters[-1], exact_offset, places=1)

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

    def test_logistic_and_linear_function(self):
        x_data = np.arange(-10, 10, 0.1)

        _ = logistic(x_data, x0=0, alpha=1)
        self.assertTrue(logistic(0, x0=0, alpha=1) == 0.5)

        _ = linear_function(x_data, 1, 2)
        self.assertTrue(linear_function(0, 1, 2) == 2)
        self.assertTrue(linear_function(3, 1, 2) == 5)


class TestGaussRamseyFunctionFit(unittest.TestCase):

    def test_fit_values(self, fig=None):
        y_data = np.array([0.6019, 0.5242, 0.3619, 0.1888, 0.1969, 0.3461, 0.5276, 0.5361,
                           0.4261, 0.28, 0.2323, 0.2992, 0.4373, 0.4803, 0.4438, 0.3392,
                           0.3061, 0.3161, 0.3976, 0.4246, 0.398, 0.3757, 0.3615, 0.3723,
                           0.3803, 0.3873, 0.3873, 0.3561, 0.37, 0.3819, 0.3834, 0.3838,
                           0.37, 0.383, 0.3573, 0.3869, 0.3838, 0.3792, 0.3757, 0.3815])
        x_data = 1e-6 * np.array([i * 1.6 / 40 for i in range(40)])

        par_fit_test, _ = fit_gauss_ramsey(x_data, y_data)

        self.assertTrue(np.abs(np.abs(par_fit_test[0]) - 0.21) < 0.1)
        self.assertTrue(np.abs(par_fit_test[-2] - (-1.255)) < 0.1)
        self.assertTrue(np.abs(par_fit_test[-1] - 0.38) < 0.1)

        x_data = np.array(
            [0.00000000e+00, 4.73333330e-07, 9.46666660e-07, 1.41999999e-06, 1.89333332e-06, 2.36666665e-06,
             2.83333338e-06, 3.30666671e-06, 3.78000004e-06, 4.25333337e-06, 4.72666670e-06,
             5.20000003e-06, 5.67333336e-06, 6.14666669e-06, 6.62000002e-06, 7.09333335e-06, 7.56666668e-06,
             8.03333296e-06, 8.50666675e-06, 8.97999962e-06, 9.45333340e-06, 9.92666628e-06, 1.04000001e-05])
        y_data = np.array(
            [0.59500003, 0.41499999, 0.12, 0.13166666, 0.35833332, 0.60500002, 0.48166665, 0.36166668, 0.28166667,
             0.26333332, 0.28999999,
             0.32666665, 0.36666667, 0.37666667, 0.38499999, 0.38833332, 0.35166666, 0.31666666, 0.34, 0.30000001,
             0.31999999, 0.33000001, 0.31666666])

        par_fit_test, results = fit_gauss_ramsey(x_data, y_data)

        self.assertAlmostEqual(par_fit_test[0], 0.33, places=1)
        self.assertAlmostEqual(par_fit_test[1] * 1e6, 3.36, places=1)
        self.assertAlmostEqual(par_fit_test[2] * 1e-6, 0.36, places=1)

    @staticmethod
    def _fit_ramsey_function(x_data, y_data, fig=None):
        fit_parameters, r = fit_gauss_ramsey(x_data, y_data)
        freq_fit = fit_parameters[2]
        t2star_fit = fit_parameters[1]
        print('Gauss Ramsey fit: f = %.3g Hz / T2* = %.2g s' % (freq_fit, t2star_fit))
        if fig:
            plot_gauss_ramsey_fit(x_data, y_data, fit_parameters, fig, initial_parameters=r['parameters initial guess'])
            fig = fig + 1
        return fit_parameters, r, fig

    def test_fit_negative_frequency_1(self, fig=None):
        data = TestData.GaussRamsey.negative_frequency_1()
        fit_parameters, r, fig = self._fit_ramsey_function(**data, fig=fig)
        freq_fit = fit_parameters[2]
        self.assertGreater(freq_fit, 0.0)

    def test_fit_negative_frequency_2(self, fig=None):
        data = TestData.GaussRamsey.negative_frequency_2()
        fit_parameters, _, fig = self._fit_ramsey_function(**data, fig=fig)
        freq_fit = fit_parameters[2]
        self.assertGreater(freq_fit, 0.0)

    def test_fit_negative_frequency_3(self, fig=None):
        data = TestData.GaussRamsey.negative_frequency_3()
        fit_parameters, _, fig = self._fit_ramsey_function(**data, fig=fig)
        freq_fit = fit_parameters[2]
        self.assertGreater(freq_fit, 0.0)

    @unittest.skip("enable once we have found a working fit on this data")
    def test_fit_high_freq_low_amp_1(self, fig=None):
        data = TestData.GaussRamsey.fit_high_freq_low_amp_1()
        fit_parameters, _, fig = self._fit_ramsey_function(**data, fig=fig)
        freq_fit = fit_parameters[2]
        amp_fit = fit_parameters[0]
        self.assertLess(freq_fit, 6.0E6)
        self.assertGreater(amp_fit, 0.2)
