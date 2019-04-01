import unittest
import numpy as np

import qtt.pgeometry as pgeometry


class TestGeometryOperations(unittest.TestCase):

    def test_pg_rotx(self):
        I = pgeometry.pg_rotx(90).dot(pgeometry.pg_rotx(-90))
        np.testing.assert_almost_equal(I, np.eye(3))

        for phi in [0, .1, np.pi, 4]:
            Rx = pgeometry.pg_rotx(phi)
            self.assertAlmostEqual(Rx[1, 1], np.cos(phi))
            self.assertAlmostEqual(Rx[2, 1], np.sin(phi))


    def test_pg_scaling(self):
        H = pgeometry.pg_scaling([1,2])
        np.testing.assert_array_equal(H, np.diag([1,2,1]))
        H = pgeometry.pg_scaling([2], [1])
        np.testing.assert_array_equal(H, np.array([[2,-1],[0,1.]]) )
        with self.assertRaises(ValueError):
            pgeometry.pg_scaling([1], [1,2])

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

if __name__ == "__main__":
    unittest.main()
