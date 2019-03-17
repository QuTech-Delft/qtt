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

    def test_pg_rotation2H(self):
        R = pgeometry.pg_rotx(.12)
        H = pgeometry.pg_rotation2H(R)
        np.testing.assert_almost_equal(R, H[:3, :3])

    def test_decomposeProjectiveTransformation(self):
        R = pgeometry.pg_rotation2H(pgeometry.pg_rotx(np.pi / 2))
        Ha, Hs, Hp, rest = pgeometry.decomposeProjectiveTransformation(R)


if __name__ == "__main__":
    """ Dummy main for testing
    """

    import unittest
    unittest.main()
