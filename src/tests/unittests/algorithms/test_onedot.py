""" Functionality for testing analysis of single quantum dots.

For more details see https://arxiv.org/abs/1603.02274

"""
# %%

import unittest
import matplotlib.pyplot as plt
from qtt.simulation.virtual_dot_array import initialize
from qtt.measurements.scans import scan2D, scanjob_t
from qtt.algorithms.onedot import onedotGetBalance, onedotGetBalanceFine, plot_onedot


class TestOneDot(unittest.TestCase):

    @staticmethod
    def one_dot(fig=None, verbose=0):
        nr_dots = 3
        station = initialize(reinit=True, nr_dots=nr_dots, maxelectrons=2, verbose=verbose)
        gates = station.gates

        gv = {'B0': -300.000, 'B1': 0.487, 'B2': -0.126, 'B3': 0.000, 'D0': 0.111, 'O1': -0.478, 'O2': 0.283, 'O3': 0.404, 'O4': 0.070,
              'O5': 0.392, 'P1': 0.436, 'P2': 0.182, 'P3': 39.570, 'SD1a': -0.160, 'SD1b': -0.022, 'SD1c': 0.425, 'bias_1': -0.312, 'bias_2': 0.063}
        gates.resetgates(gv, gv, verbose=verbose)

        start = -250
        scanjob = scanjob_t({'sweepdata': dict(
            {'param': 'B0', 'start': start, 'end': start + 200, 'step': 4., 'wait_time': 0.}), 'minstrument': ['keithley3.amplitude']})
        scanjob['stepdata'] = dict({'param': 'B1', 'start': start, 'end': start + 200, 'step': 5.})
        data = scan2D(station, scanjob, verbose=verbose)

        x = onedotGetBalance(dataset=data, verbose=verbose, fig=fig)
        results = x[0]
        _, _ = onedotGetBalanceFine(impixel=None, dd=data, verbose=verbose, fig=fig)
        plot_onedot(results, ds=data, fig=fig, verbose=verbose)

    def test_one_dot(self, fig=None, verbose=0):
        plt.interactive(True)
        self.one_dot(fig, verbose)
        plt.show()
        plt.close('all')
