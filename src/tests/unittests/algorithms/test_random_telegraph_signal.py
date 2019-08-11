""" Functionality to test analyse random telegraph signals."""

import unittest
import numpy as np
import matplotlib.pyplot as plt
from qtt.algorithms.random_telegraph_signal import tunnelrates_RTS, FittingException, generate_RTS_signal, \
    _create_integer_histogram
import warnings


class TestRandomTelegraphSignal(unittest.TestCase):

    def test_RTS(self, fig=None, verbose=0):
        data = np.random.rand(10000, )
        try:
            _ = tunnelrates_RTS(data)
            raise Exception('no samplerate available')
        except Exception as ex:
            # exception is good, since no samplerate was provided
            self.assertTrue('samplerate is None' in str(ex))
        try:
            _ = tunnelrates_RTS(data, samplerate=10e6)
            raise Exception('data should not fit to RTS')
        except FittingException as ex:
            # fitting exception is good, since data is random
            pass

        data = generate_RTS_signal(100, std_gaussian_noise=0, uniform_noise=.1)
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

    def test_RTS_fractions(self, fig=None, verbose=0):
        data = np.sin(np.arange(0, 1000, .1)) > .6
        data = data + (np.random.rand(data.size) - .5) / 10
        tunnelrate_dn, tunnelrate_up, rts_results = tunnelrates_RTS(data, samplerate=1e6, min_duration=1, max_sep=40,
                                                                    fig=None, verbose=0)
        self.assertAlmostEqual(tunnelrate_dn, 1000, -2)
        self.assertAlmostEqual(tunnelrate_up, 1000, -2)
        self.assertAlmostEqual(rts_results['fraction_down'], 0.7047, 1)
        self.assertEqual(rts_results['fraction_down'], 1 - rts_results['fraction_up'])

    def test_create_integer_histogram(self):
        input_data = np.array([2, 3, 4])
        counts, bins, bin_size = _create_integer_histogram(input_data)
        self.assertEqual(input_data.size, np.sum(counts))

        input_data = np.array([2.])
        counts, bins, bin_size = _create_integer_histogram(input_data)
        self.assertEqual(input_data.size, np.sum(counts))
        self.assertEqual(len(bins), 2)
