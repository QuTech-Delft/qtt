import unittest
import numpy as np
from qtt.algorithms.functions import fit_gauss_ramsey, plot_gauss_ramsey_fit


class TestFunctions(unittest.TestCase):

    def test_fit_gauss_ramsey(self, fig=100):
        y_data = np.array([0.6019, 0.5242, 0.3619, 0.1888, 0.1969, 0.3461, 0.5276, 0.5361,
                           0.4261, 0.28, 0.2323, 0.2992, 0.4373, 0.4803, 0.4438, 0.3392,
                           0.3061, 0.3161, 0.3976, 0.4246, 0.398, 0.3757, 0.3615, 0.3723,
                           0.3803, 0.3873, 0.3873, 0.3561, 0.37, 0.3819, 0.3834, 0.3838,
                           0.37, 0.383, 0.3573, 0.3869, 0.3838, 0.3792, 0.3757, 0.3815])
        x_data = np.array([i * 1.6 / 40 for i in range(40)])

        par_fit_test, _ = fit_gauss_ramsey(x_data * 1e-6, y_data)

        plot_gauss_ramsey_fit(x_data, y_data, par_fit_test, fig=fig)

