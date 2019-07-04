
import unittest
import numpy as np
from skimage.feature import peak_local_max
from qtt.algorithms.generic import subpixelmax, rescaleImage, smoothImage, boxcar_filter
import matplotlib.pyplot as plt


class TestGeneric(unittest.TestCase):

    @staticmethod
    def test_subpixel(fig=100):
        np.random.seed(2019)
        ims = np.random.rand(40,)**2 + 1e1
        smooth_ims = smoothImage(ims)

        mpos = peak_local_max(smooth_ims, min_distance=3).flatten()
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

    def test_rescale_image(self, fig=100):
        np.random.seed(2019)
        im = np.random.rand(300, 600)
        rescaled, _, s = rescaleImage(im, [0, im.shape[1] - 1, 0,
                                      im.shape[0] - 1], mvx=4, verbose=0, fig=fig)
        self.assertTrue(rescaled.size, 45000)
        self.assertSequenceEqual(s, (4, 1.0, 0.25, 1.0))
        plt.close('all')


class TestBoxcarFilter(unittest.TestCase):

    _kernel_size_0D: tuple = ()
    _kernel_size_1D = (3,)
    _kernel_size_2D = (3, 3)
    _kernel_size_3D = (3, 3, 5)

    _signal_1D = [1.25, 4.52, 46.36]
    _signal_2D = [[1.25, 4.52, 46.36],
                 [3.45, 25.66, 6234.6]]

    _known_inputs_outputs_1D = [
        ([],
         []),

        ([5.],
         [5.]),

        (np.array([4., 6.]),
         np.array([14., 16.]) / 3.),

        (np.ones(5),
         np.ones(5)),

        (np.array([1., 2., 1.]),
         np.array([4., 4., 4.]) / 3.),

        (np.array([0., 1., 2., 3.]),
         np.array([1., 3., 6., 8.]) / 3.),

        (np.array([-1,  1., 2., 3., -3.]),
         np.array([-1., 2., 6., 2., -3.]) / 3.),

        (np.array([-1,   1., -1., 1., -1., 1.]),
         np.array([-1., -1.,  1, -1.,  1., 1.]) / 3.),

        (np.array([-0.5, 1.5, 2.5, 3.5, -2.5]),
         np.array([ 0.5, 3.5, 7.5, 3.5, -1.5]) / 3.),
    ]

    _known_inputs_outputs_2D = [
        ([[],
          []],
         [[],
          []]),

        ([[5.],
          [5.]],
         [[5.],
          [5.]]),

        (np.array([[4., 6.],
                   [4., 6.]]),
         np.array([[14., 16.],
                   [14., 16.]]) / 3.),

        (np.ones((5, 5)),
         np.ones((5, 5))),

        (np.array([[1., 2., 1.],
                   [1., 2., 1.]]),
         np.array([[4., 4., 4.],
                   [4., 4., 4.]]) / 3.),

        (np.array([[0., 1., 2., 3.],
                   [0., 1., 2., 3.]]),
         np.array([[1., 3., 6., 8.],
                   [1., 3., 6., 8.]]) / 3.),

        (np.array([[-1,  1., 2., 3., -3.],
                   [-1,  1., 2., 3., -3.]]),
         np.array([[-1., 2., 6., 2., -3.],
                   [-1., 2., 6., 2., -3.]]) / 3.),

        (np.array([[-1,   1., -1., 1., -1., 1.],
                  [-1,   1., -1., 1., -1., 1.]]),
         np.array([[-1., -1.,  1, -1.,  1., 1.],
                   [-1., -1.,  1, -1.,  1., 1.]]) / 3.),

        (np.array([[-0.5, 1.5, 2.5, 3.5, -2.5],
                   [-0.5, 1.5, 2.5, 3.5, -2.5]]),
         np.array([[ 0.5, 3.5, 7.5, 3.5, -1.5],
                   [ 0.5, 3.5, 7.5, 3.5, -1.5]]) / 3.),
    ]


    def assertFilterOutput(self, signal, kernel, out_exp):
        """Assert that signal generates output close to expectation"""
        with self.subTest(signal=signal):
            out_fnd = boxcar_filter(signal, kernel)
            msg = ''
            try:
                np.testing.assert_allclose(out_fnd, out_exp)
            except AssertionError as e:
                msg = '\nsignal: %s\ngives wrong filtered output:' % (signal,)
                msg += str(e).replace('x:', '  actual:').replace('y:', 'expected:')

            if msg:
                self.fail(msg)


    def test_float_inputs(self):
        kernel_1D = TestBoxcarFilter._kernel_size_1D
        for signal, out_exp in TestBoxcarFilter._known_inputs_outputs_1D:
            self.assertFilterOutput(signal, kernel_1D, out_exp)

        kernel_2D = TestBoxcarFilter._kernel_size_2D
        for signal, out_exp in TestBoxcarFilter._known_inputs_outputs_2D:
            self.assertFilterOutput(signal, kernel_2D, out_exp)

    def test_int_inputs(self):
        kernel_1D = TestBoxcarFilter._kernel_size_1D
        for signal, out_exp in TestBoxcarFilter._known_inputs_outputs_1D:
            int_signal = np.array(signal, dtype=int)
            try:
                np.testing.assert_allclose(int_signal, signal)
            except AssertionError:
                continue

            self.assertFilterOutput(int_signal, kernel_1D, out_exp)

        kernel_2D = TestBoxcarFilter._kernel_size_2D
        for signal, out_exp in TestBoxcarFilter._known_inputs_outputs_2D:
            int_signal = np.array(signal, dtype=int)
            try:
                np.testing.assert_allclose(int_signal, signal)
            except AssertionError:
                continue

            self.assertFilterOutput(int_signal, kernel_2D, out_exp)




    def test_wrong_kernel_dimensions(self):
        """
        If the number of dimensions in the kernel is not equal to the number of dimensions
        in the signal, an exception should be raised
        Test kernels: 0D, 1D, 2D, 3D
        """
        signal_1D = TestBoxcarFilter._signal_1D
        signal_2D = TestBoxcarFilter._signal_2D
        kernel_0D = TestBoxcarFilter._kernel_size_0D
        kernel_1D = TestBoxcarFilter._kernel_size_1D
        kernel_2D = TestBoxcarFilter._kernel_size_2D
        kernel_3D = TestBoxcarFilter._kernel_size_3D
        self.assertRaises(RuntimeError, boxcar_filter, signal_1D, kernel_0D)
        self.assertRaises(RuntimeError, boxcar_filter, signal_1D, kernel_2D)
        self.assertRaises(RuntimeError, boxcar_filter, signal_2D, kernel_0D)
        self.assertRaises(RuntimeError, boxcar_filter, signal_2D, kernel_1D)
        self.assertRaises(RuntimeError, boxcar_filter, signal_2D, kernel_3D)


    def test_invalid_kernel_size(self):
        """
        If any of the dimensions of the kernel has non-positive size, an exception should be raised
        """
        signal_1D = TestBoxcarFilter._signal_1D
        signal_2D = TestBoxcarFilter._signal_2D
        self.assertRaises(RuntimeError, boxcar_filter, signal_1D, (0,))
        self.assertRaises(RuntimeError, boxcar_filter, signal_1D, (-1,))
        self.assertRaises(RuntimeError, boxcar_filter, signal_2D, (3, 0))
        self.assertRaises(RuntimeError, boxcar_filter, signal_2D, (5, -1))



