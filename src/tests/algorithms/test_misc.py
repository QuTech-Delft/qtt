""" Testing misc algorithms

"""

# %%
import unittest
from qtt.algorithms.misc import polyfit2d, polyval2d, point_in_poly
import numpy as np


class TestMis(unittest.TestCase):

    def test_polyfit2d(self):
        np.random.seed(2019)
        x = np.arange(10., 20)
        y = np.arange(20., 30)
        z = np.random.rand(10, 10)
        p = polyfit2d(x, y, z)
        self.assertEqual(p.size, 160)
        self.assertAlmostEqual(p.min(), -271.1967198595069, 6)
        self.assertAlmostEqual(p.max(), 178.449535954755, 6)
        self.assertAlmostEqual(p.sum(), -542.5819643077925, 3)

    def test_polyval2d(self):
        np.random.seed(2019)
        x = np.arange(10., 20)
        y = np.arange(20., 30)
        z = np.random.rand(10, 10)
        p = polyfit2d(x, y, z)
        q = polyval2d(x, y, p)
        self.assertEqual(q.size, 10)
        self.assertAlmostEqual(q.min(), 0.04216012929100543, 6)
        self.assertAlmostEqual(q.max(), 0.9130578654339843, 6)

    def test_point_in_poly(self):
        x = 1.0
        y = 1.0
        poly = [[0, 0], [1, 0], [1, 1], [0, 2]]
        inside = point_in_poly(x, y, poly)
        self.assertTrue(inside)
        x = 0.5
        y = 2.0
        inside = point_in_poly(x, y, poly)
        self.assertFalse(inside)




