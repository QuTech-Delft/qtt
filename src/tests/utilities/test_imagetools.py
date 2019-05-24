
import unittest
import numpy as np
from qtt.utilities.imagetools import fitBackground, cleanSensingImage, semiLine, lineSegment


class TestImageTools(unittest.TestCase):

    def test_fit_background(self):
        np.random.seed(2019)
        im = np.random.rand(200, 100)
        bg = fitBackground(im, verbose=0)
        bgo = fitBackground(im, verbose=0, removeoutliers=True)
        c = cleanSensingImage(im)

        self.assertEqual(bg.size, 20000)
        self.assertAlmostEqual(bg.min(), 4.180688953684698e-06)
        self.assertAlmostEqual(bg.max(), 0.6620333371459401)
        self.assertAlmostEqual(bg.sum(), 9725.006056739836)
        self.assertEqual(bgo.size, 20000)
        self.assertAlmostEqual(bgo.min(), 3.6926296812183113e-06)
        self.assertAlmostEqual(bgo.max(), 0.6424973079407454)
        self.assertAlmostEqual(bgo.sum(), 9669.358784435484)
        self.assertEqual(c.size, 20000)
        self.assertAlmostEqual(c.min(), -0.9865077626697246)
        self.assertAlmostEqual(c.max(), 0.9970046055805283)
        self.assertAlmostEqual(c.sum(), -0.786837730323506)

    def test_semi_line(self):
        im = np.zeros((300, 400))
        semi_line = semiLine(im, [100, 200], 10, w=12, l=300)

        self.assertEqual(semi_line.size, 120000)
        self.assertAlmostEqual(semi_line.min(), -8.742276804696303e-06)
        self.assertAlmostEqual(semi_line.max(), 199.9999542236328)
        self.assertAlmostEqual(semi_line.sum(), 183001.1215185296)

        line_segment = lineSegment(semi_line, [5, 5], [50, 100])
        self.assertEqual(line_segment.size, 120000)
        self.assertAlmostEqual(line_segment.min(), -1.7484456293459516e-05)
        self.assertAlmostEqual(line_segment.max(), 199.9999542232738)
        self.assertAlmostEqual(line_segment.sum(), 209757.99804017632)
