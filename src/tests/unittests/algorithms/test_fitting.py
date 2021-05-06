""" Test fitting of Fermi-Dirac distributions."""

import unittest
import numpy as np
import matplotlib.pyplot as plt
import qtt
from qtt.algorithms.functions import FermiLinear, linear_function, double_gaussian, gaussian, sine
from qtt.algorithms.fitting import initFermiLinear, _estimate_fermi_model_center_amplitude, fitFermiLinear,\
    fit_double_gaussian, refit_double_gaussian, fit_gaussian, fit_sine


class TestSineFitting(unittest.TestCase):

    def test_fit_sine(self):
        x_data = np.linspace(0, 6, 30)
        amplitude = 2
        frequency = 1.3
        phase = .1
        offset = .5
        y_data = sine(x_data, amplitude, frequency, phase, offset)+.08*(np.random.rand(x_data.size))

        fit_parameters, _results = fit_sine(x_data, y_data)
        self.assertAlmostEqual(fit_parameters[0], amplitude, places=1)
        self.assertAlmostEqual(fit_parameters[1], frequency, places=1)
        self.assertAlmostEqual(np.mod(fit_parameters[2], 2*np.pi), phase, places=1)
        self.assertAlmostEqual(fit_parameters[3], offset, places=1)

    def test_fit_sine_regression(self):
        y_data = np.array([0.60000002, 0.38999999, 0.25, 0.05, 0.,
                           0., 0., 0.15000001, 0.09, 0.44,
                           0.56, 0.76999998, 0.88999999, 0.93000001, 0.99000001,
                           1., 0.88999999, 0.86000001, 0.74000001, 0.38])

        x_data = np.array([0., 0.33069396, 0.66138792, 0.99208188, 1.32277584,
                           1.6534698, 1.98416376, 2.31485772, 2.64555168, 2.97624564,
                           3.3069396, 3.63763356, 3.96832752, 4.29902172, 4.62971544,
                           4.96040964, 5.29110336, 5.62179756, 5.95249128, 6.28318548])
        fit_parameters, results = fit_sine(x_data, y_data)

        phase = np.mod(results['fitted_parameter_dictionary']['phase'], 2*np.pi)
        self.assertAlmostEqual(results['fitted_parameter_dictionary']['amplitude'], 0.524, places=2)
        self.assertAlmostEqual(results['fitted_parameter_dictionary']['frequency'], 0.1643, places=1)
        self.assertAlmostEqual(phase, 2.98, places=1)


class TestDoubleGaussianFitting(unittest.TestCase):

    def test_refit_double_gaussian(self):
        dataset = qtt.data.load_example_dataset('double_gaussian_dataset.json')
        x_data = np.array(dataset.signal)
        y_data = np.array(dataset.counts)

        _, result_dict = fit_double_gaussian(x_data, y_data)
        result_dict = refit_double_gaussian(result_dict, x_data, y_data)

        self.assertAlmostEqual(result_dict['left'][0], -1.23, places=1)
        self.assertAlmostEqual(result_dict['right'][0], -0.77, places=1)
        self.assertAlmostEqual(result_dict['left'][2], 162.8, places=0)
        self.assertAlmostEqual(result_dict['right'][2], 1971, places=-1)

    def test_fit_double_gaussian(self):
        x_data = np.arange(-4, 4, .05)
        initial_parameters = [10, 20, 1, 1, -2, 2]
        y_data = double_gaussian(x_data, initial_parameters)

        fitted_parameters, _ = fit_double_gaussian(x_data, y_data)
        parameter_diff = np.abs(fitted_parameters - initial_parameters)
        self.assertTrue(np.all(parameter_diff < 1e-3))


class TestGaussianFitting(unittest.TestCase):

    def test_fit_gaussian_no_offset(self):
        x_data = np.linspace(0, 10, 100)
        gauss_data = gaussian(x_data, mean=4, std=1, amplitude=5)
        noise = np.random.rand(100) - .5
        parameters, result_dict = fit_gaussian(x_data=x_data, y_data=(gauss_data + noise), estimate_offset=0)
        np.testing.assert_array_almost_equal(result_dict['fitted_parameters'], np.array([4, 1, 5.]), decimal=1)
        np.testing.assert_array_almost_equal(parameters, result_dict['fitted_parameters'])
        self.assertTrue(result_dict['reduced_chi_squared'] < .2)

    def test_fit_gaussian(self):
        x_data = np.linspace(0, 10, 100)
        gauss_data = 0.1 + gaussian(x_data, mean=4, std=1, amplitude=5)
        noise = np.random.rand(100) - .5
        [mean, s, amplitude, offset], _ = fit_gaussian(x_data=x_data, y_data=(gauss_data + noise))
        self.assertTrue(3.5 < mean < 4.5)
        self.assertTrue(0.5 < s < 1.5)
        self.assertTrue(4.5 < amplitude < 5.5)
        self.assertAlmostEqual(offset, 0.1, places=0)


class TestFermiFitting(unittest.TestCase):

    def test_initial_estimate_fermi_linear(self, fig=None):
        expected_parameters = [0.01000295, 0.51806569, -4.88800525, 0.12838861, 0.25382811]
        x_data = np.arange(-20, 10, 0.1)
        y_data = FermiLinear(x_data, *expected_parameters)
        y_data += 0.005 * np.random.rand(y_data.size)

        linear_part, fermi_part = initFermiLinear(x_data, y_data, fig=fig)

        ylin = linear_function(x_data, *linear_part)
        yr = y_data - ylin

        cc, A = _estimate_fermi_model_center_amplitude(x_data, yr, fig=fig)
        np.testing.assert_almost_equal(cc, expected_parameters[2], decimal=1)
        np.testing.assert_almost_equal(A, expected_parameters[3], decimal=1)
        self.assertTrue(fermi_part is not None)
        plt.close('all')

    def test_fit_fermi_linear(self, fig=None, verbose=0):
        expected_parameters = [0.01000295, 0.51806569, -4.88800525, 0.12838861, 0.25382811]
        x_data = np.arange(-20, 10, 0.1)
        y_data = FermiLinear(x_data, *expected_parameters)
        y_data += 0.005 * np.random.rand(y_data.size)

        actual_parameters, _ = fitFermiLinear(x_data, y_data, verbose=verbose, fig=fig, use_lmfit=False)
        absolute_difference_parameters = np.abs(actual_parameters - expected_parameters)

        y_data_fitted = FermiLinear(x_data, *actual_parameters)
        max_difference = np.max(np.abs(y_data_fitted - y_data))

        if verbose:
            print('expected: %s' % expected_parameters)
            print('fitted:   %s' % actual_parameters)
            print('temperature: %.2f' % (actual_parameters[-1]))
            print('max diff parameters: %.2f' % (absolute_difference_parameters.max()))
            print('max diff values: %.4f' % (max_difference.max()))

        self.assertTrue(np.all(max_difference < 1.0e-2))
        self.assertTrue(np.all(absolute_difference_parameters < 0.6))

        try:
            import lmfit
            have_lmfit = True
        except ImportError:
            have_lmfit = False

        if have_lmfit:
            self.assertIsNotNone(lmfit.__file__)
            actual_parameters, _ = fitFermiLinear(x_data, y_data, verbose=1, fig=fig, use_lmfit=True)
            absolute_difference_parameters = np.abs(actual_parameters - expected_parameters)
            self.assertTrue(np.all(absolute_difference_parameters < 0.1))

        # test with data from F006
        x_data = np.array([-20., -19.9, -19.8, -19.7, -19.6, -19.5, -19.4, -19.3, -19.2,
                           -19.1, -19., -18.9, -18.8, -18.7, -18.6, -18.5, -18.4, -18.3,
                           -18.2, -18.1, -18., -17.9, -17.8, -17.7, -17.6, -17.5, -17.4,
                           -17.3, -17.2, -17.1, -17., -16.9, -16.8, -16.7, -16.6, -16.5,
                           -16.4, -16.3, -16.2, -16.1, -16., -15.9, -15.8, -15.7, -15.6,
                           -15.5, -15.4, -15.3, -15.2, -15.1, -15., -14.9, -14.8, -14.7,
                           -14.6, -14.5, -14.4, -14.3, -14.2, -14.1])
        y_data = np.array([0.03055045, 0.0311075, 0.03098561, 0.03033496, 0.03006341,
                           0.03072266, 0.03183486, 0.03170599, 0.03199145, 0.03224666,
                           0.03164276, 0.03156053, 0.03133487, 0.03184649, 0.03224385,
                           0.03207413, 0.03196082, 0.03229934, 0.03158735, 0.03120681,
                           0.03119833, 0.03220412, 0.03185901, 0.03124884, 0.03129008,
                           0.0314923, 0.0315841, 0.0313667, 0.03115382, 0.03069049,
                           0.03058055, 0.02923863, 0.02789339, 0.02437544, 0.01896179,
                           0.01776424, 0.01175409, 0.01074043, 0.00950811, 0.0074723,
                           0.0060949, 0.00575982, 0.00501728, 0.00490061, 0.00465821,
                           0.00440039, 0.00434098, 0.00429608, 0.00421024, 0.0042945,
                           0.0042552, 0.00433429, 0.00440945, 0.00446915, 0.00446351,
                           0.00439317, 0.00447768, 0.0044295, 0.00450926, 0.0045605])

        figx = fig if fig is None else fig + 100
        linear_part, fermi_part = initFermiLinear(x_data, y_data, fig=figx)
        np.testing.assert_almost_equal(linear_part, [0, 0], decimal=1)
        np.testing.assert_almost_equal(fermi_part, [-16.7, 0.02755, 0.01731], decimal=2)
        plt.close('all')
