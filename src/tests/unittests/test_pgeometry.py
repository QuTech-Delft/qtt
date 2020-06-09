import unittest
import numpy as np
import matplotlib.pyplot as plt
import qtt.pgeometry as pgeometry
from qtt.pgeometry import point_in_polygon, points_in_polygon


class TestPGeometry(unittest.TestCase):

    @staticmethod
    def test_robust_cost():
        x = np.array([0, 1, 2, 3, 4, 5])
        _ = pgeometry.robustCost(x, 2)
        _ = pgeometry.robustCost(x, 'auto')


class TestGeometryOperations(unittest.TestCase):

    def test_pg_rotx(self):
        I = pgeometry.pg_rotx(90).dot(pgeometry.pg_rotx(-90))
        np.testing.assert_almost_equal(I, np.eye(3))

        for phi in [0, .1, np.pi, 4]:
            Rx = pgeometry.pg_rotx(phi)
            self.assertAlmostEqual(Rx[1, 1], np.cos(phi))
            self.assertAlmostEqual(Rx[2, 1], np.sin(phi))

    def test_pg_scaling(self):
        H = pgeometry.pg_scaling([1, 2])
        np.testing.assert_array_equal(H, np.diag([1, 2, 1]))
        H = pgeometry.pg_scaling([2], [1])
        np.testing.assert_array_equal(H, np.array([[2, -1], [0, 1.]]))
        with self.assertRaises(ValueError):
            pgeometry.pg_scaling([1], [1, 2])

    def test_pg_rotation2H(self):
        R = pgeometry.pg_rotx(0.12)
        H = pgeometry.pg_rotation2H(R)
        np.testing.assert_almost_equal(R, H[:3, :3])
        self.assertIsNotNone(H)

    def test_decomposeProjectiveTransformation(self):
        R = pgeometry.pg_rotation2H(pgeometry.pg_rotx(np.pi / 2))
        Ha, Hs, Hp, _ = pgeometry.decomposeProjectiveTransformation(R)
        self.assertIsInstance(Ha, np.ndarray)
        np.testing.assert_array_almost_equal(Hs @ Ha @ Hp, R)

        R = pgeometry.pg_rotation2H(pgeometry.pg_rotx(.012))
        Ha, Hs, Hp, _ = pgeometry.decomposeProjectiveTransformation(R)
        np.testing.assert_array_almost_equal(Hs @ Ha @ Hp, R)

    def test_directionMean(self):
        directions = [[1, 0], [1, .1], [1, -.1]]
        angle = pgeometry.directionMean(directions)
        angle = np.mod(angle, np.pi)
        self.assertAlmostEqual(angle, np.pi / 2)

        directions = [[1, 0], [-1, 0]]
        angle = pgeometry.directionMean(directions)
        angle = np.mod(angle, np.pi)
        self.assertAlmostEqual(angle, np.pi / 2)


class TestPolygonGeometry(unittest.TestCase):

    def assertEmptyPolygon(self, polygon):
        self.assertTrue(len(polygon) == 0)

    def test_polyintersect(self):
        x1 = np.array([(0, 0), (1, 1), (1, 0)])
        x2 = np.array([(1, 0), (1.5, 1.5), (.5, 0.5)])
        intersection_polygon = pgeometry.polyintersect(x1, x2)
        self.assertEqual(4, len(intersection_polygon))
        self.assertEqual(0.25, np.abs(pgeometry.polyarea(intersection_polygon)))

    def test_polyintersect_empty_intersection(self):
        x1 = np.array([(0, 0), (1, 1), (1, 0)])
        x2 = 100 + np.array([(1, 0), (1.5, 1.5), (.5, 0.5)])
        intersection_polygon = pgeometry.polyintersect(x1, x2)
        self.assertEmptyPolygon(intersection_polygon)

    def test_polyintersect_identical_polygons(self):
        x1 = np.array([(0, 0), (1, 1), (1, 0)])
        intersection_polygon = pgeometry.polyintersect(x1, x1)
        for pt in x1:
            self.assertIn(pt, intersection_polygon)

    def test_polyintersect_degenerate_polygon(self):
        x1 = np.array([(0, 0)])
        x2 = np.array([(1, 0), (1.5, 1.5)])

        with self.assertRaises(ValueError):
            intersection_polygon = pgeometry.polyintersect(x1, x2)

    def test_multi_polygon_intersection(self):
        delta = .42
        shift = 1.7
        x1 = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)]) + np.array([shift, 0])
        x2 = np.array([(delta, 0), (5, 5), (-5, 5), (-delta, 0), (-5, -5), (5, -5), (delta, 0)])

        with self.assertRaises(ValueError):
            intersection_polygon = pgeometry.polyintersect(x1, x2)

    def test_non_convex_intersection(self):
        delta = .5
        x1 = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)])
        x2 = np.array([(delta, 0), (5, 5), (-5, 5), (-delta, 0), (-5, -5), (5, -5), (delta, 0)])
        intersection_polygon = pgeometry.polyintersect(x1, x2)

        self.assertEqual(intersection_polygon.shape, (11, 2))
        self.assertEqual(np.abs(pgeometry.polyarea(intersection_polygon)), 31 / 9)

    def test_geometry(self, fig=None):
        im = np.zeros((200, 100, 3))
        sub_im = np.ones((40, 30,))
        im = pgeometry.setregion(im, sub_im, [0, 0])
        im = np.zeros((200, 100, 3))
        sub_im = np.ones((40, 30,))
        im = pgeometry.setregion(im, sub_im, [95, 0], clip=True)
        if fig:
            plt.figure(fig)
            plt.clf()
            plt.imshow(im, interpolation='nearest')
        self.assertIsInstance(im, np.ndarray)

    def test_intersect2lines(self):
        p1 = np.array([[0, 0]])
        p2 = np.array([[1, 0]])
        p3 = np.array([[1, 1]])
        p4 = np.array([[2, 2]])

        line1 = pgeometry.fitPlane(np.vstack((p1, p2)))
        line2 = pgeometry.fitPlane(np.vstack((p3, p4)))

        a = pgeometry.intersect2lines(line1, line2)
        pt = pgeometry.dehom(a)
        self.assertIsNotNone(pt)
        np.testing.assert_almost_equal(pt, 0)

    def test_polygon_functions(self):
        pp = np.array([[0, 0], [4, 0], [0, 4]])
        self.assertTrue(point_in_polygon([1, 1], pp) == 1)
        self.assertTrue(point_in_polygon([-1, 1], pp) == -1)
        self.assertTrue(np.all(points_in_polygon(np.array([[-1, 1], [1, 1], [.5, .5]]), pp) == np.array([-1, 1, 1])))
