""" Functionality to test analyse random telegraph signals."""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from qtt.algorithms.random_telegraph_signal import tunnelrates_RTS, generate_RTS_signal
import warnings


class TestRandomTelegraphSignal(unittest.TestCase):

    def test_RTS(self, fig=100, verbose=2):
        data = generate_RTS_signal(100, std_gaussian_noise=0.1, uniform_noise=.1)

        samplerate = 2e6
        data = generate_RTS_signal(100000, std_gaussian_noise=0.1, rate_up=10e3, rate_down=20e3, samplerate=samplerate)

        with warnings.catch_warnings():  # catch any warnings
            warnings.simplefilter("ignore")
            tunnelrate_dn, tunnelrate_up, parameters = tunnelrates_RTS(data, samplerate=samplerate, fig=fig,
                                                                       verbose=verbose)

            self.assertTrue(parameters['up_segments']['mean'] > 0)
            self.assertTrue(parameters['down_segments']['mean'] > 0)

        samplerate = 1e6
        rate_up = 200e3
        rate_down = 20e3
        data = generate_RTS_signal(100000, std_gaussian_noise=0.01, rate_up=rate_up,
                                   rate_down=rate_down, samplerate=samplerate)

        tunnelrate_dn, tunnelrate_up, _ = tunnelrates_RTS(data, samplerate=samplerate, min_sep=1.0, max_sep=2222,
                                                          min_duration=1, num_bins=40, fig=fig, verbose=verbose)

        self.assertTrue(np.abs(tunnelrate_dn - rate_up * 1e-3) < 100)
        self.assertTrue(np.abs(tunnelrate_up - rate_down * 1e-3) < 10)
        plt.close('all')