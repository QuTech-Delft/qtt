
import unittest
import numpy as np
from skimage.feature import peak_local_max
from qtt.algorithms.generic import subpixelmax, rescaleImage, smoothImage
import matplotlib.pyplot as plt


class TestGeneric(unittest.TestCase):

    def test_subpixel(self, fig=None):
        np.random.seed(2019)
        ims = np.random.rand(40,)**2 + 1e1
        smooth_ims = smoothImage(ims)

        mpos = peak_local_max(smooth_ims, min_distance=3).flatten()
        # make sure the local max are always ordered the same way (descending)
        mpos[::-1].sort()
        self.assertListEqual(list(mpos), [34, 20, 14, 7])
        subpos, subval = subpixelmax(smooth_ims, mpos)

        np.testing.assert_array_almost_equal(subpos, np.array([34.48945739, 19.7971219, 14.04429215, 7.17665281]),
                                             decimal=8)
        np.testing.assert_array_almost_equal(subval, np.array([10.75670888, 10.46787185, 10.59470934, 10.70051573]),
                                             decimal=8)
        if fig:
            plt.figure(fig)
            plt.clf()
            plt.plot(np.arange(ims.size), ims, '.:r', label='data points')

            plt.plot(mpos, ims[mpos], 'om', label='integer maxima')
            plt.plot(subpos, subval, '.g', markersize=15, label='subpixel maxima')
            plt.legend(numpoints=1)
            plt.close('all')

    def test_rescale_image(self, fig=None):
        np.random.seed(2019)
        im = np.random.rand(300, 600)
        rescaled, _, s = rescaleImage(im, [0, im.shape[1] - 1, 0,
                                      im.shape[0] - 1], mvx=4, verbose=0, fig=fig)
        self.assertTrue(rescaled.size, 45000)
        self.assertSequenceEqual(s, (4, 1.0, 0.25, 1.0))
        plt.close('all')
