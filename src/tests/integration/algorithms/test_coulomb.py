""" Test functions to fit and analyse Coulomb peaks."""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from qtt.algorithms.functions import gaussian
from qtt.algorithms.coulomb import coulombPeaks



class TestCoulomb(unittest.TestCase):

    def setUp(self):
        x_data = np.arange(0, 100)
        y_data = np.random.rand(x_data.size)
        y_data += gaussian(x_data, 30, 6, 30)
        y_data += gaussian(x_data, 70, 8, 70)
        self.example_data = (x_data, y_data)

    def test_coulomb_peaks(self, fig=100, verbose=1):
        x_data, y_data = self.example_data

        peaks = coulombPeaks(x_data, y_data, verbose=verbose, sampling_rate=1, fig=fig)
        self.assertTrue(len(peaks) == 2)
        self.assertTrue(np.abs(peaks[0]['x'] - 70) < 2)
        self.assertTrue(np.abs(peaks[1]['x'] - 30) < 2)
        plt.close('all')
