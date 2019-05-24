""" Test functionality to analyse bias triangles

"""

import numpy as np
import unittest
from qtt.algorithms.bias_triangles import lever_arm


class TestBiasTriangles(unittest.TestCase):

    def test_lever_arm(self):
        lever_arm_fit = {
            'clicked_points': np.array([[24., 38., 40.], [135., 128., 111.]]),
            'distance': 15.0,
            'intersection_point': np.array([[40.4], [127.]])
        }

        test_lever_arm = lever_arm(-800, lever_arm_fit)
        self.assertAlmostEqual(test_lever_arm, 53.3, 1)
