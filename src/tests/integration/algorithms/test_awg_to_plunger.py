""" Functionality to test determine the awg_to_plunger ratio."""

# %% Load packages
import numpy as np
import matplotlib.pyplot as plt
import qtt.pgeometry
from qtt.utilities.imagetools import semiLine
from qtt.data import makeDataSet2Dplain
from qtt.algorithms.awg_to_plunger import analyse_awg_to_plunger
import unittest


class TestAwgToPlunger(unittest.TestCase):

    def awg_to_plunger(self, fig=100):
        """ Plot results of awg_to_plunger calibration check?

        Args:
            fig (str, int or None): default None. Name of figure to plot in, if None not plotted.

        """
        x = np.arange(0, 80, 1.0).astype(np.float32)
        y = np.arange(0, 60).astype(np.float32)
        z = np.meshgrid(x, y)
        z = 0.01 * z[0].astype(np.uint8)
        input_angle = np.deg2rad(-35)
        semiLine(z, np.array([[0], [y.max()]]), input_angle, w=2.2, l=30, H=0.52)
        ds = makeDataSet2Dplain('x', x, 'y', y, 'z', z)
        result = {'dataset': ds, 'type': 'awg_to_plunger'}
        r = analyse_awg_to_plunger(result, method='hough', fig=fig)
        output_angle = r['angle']

        if fig is not None:
            print(r)
            print('angle input %.3f: output angle %s' % (input_angle, str(output_angle)))

        d = qtt.pgeometry.angleDiff(input_angle, (-np.pi / 2 - output_angle))
        self.assertTrue(np.abs(d) < np.deg2rad(5))

    def test_awg_to_plunger(self, fig=100):
        plt.interactive(False)
        self.awg_to_plunger(fig=fig)
        plt.close('all')
