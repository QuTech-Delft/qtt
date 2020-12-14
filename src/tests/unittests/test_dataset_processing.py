import unittest

import numpy as np
from qcodes.tests.legacy.data_mocks import DataSet1D, DataSet2D

from qtt.dataset_processing import slice_dataset, process_dataarray, average_dataset, dataset_dimension, \
    average_multirow_dataset, resample_dataset


class TestDataProcessing(unittest.TestCase):

    def test_slice_dataset(self):
        dataset = DataSet1D()
        dataset_sliced = slice_dataset(dataset, [2, 5])
        self.assertEqual(dataset_sliced.default_parameter_array().shape, (3,))

        dataset = DataSet2D()
        dataset_sliced = slice_dataset(dataset, [3, 5], axis=0)
        np.testing.assert_array_equal(dataset_sliced.z, np.array([[9, 10, 13, 18], [16, 17, 20, 25]]))

        dataset_sliced = slice_dataset(dataset, [1, 5], axis=1)
        np.testing.assert_array_equal(dataset_sliced.z, np.array(
            [[1, 4], [2, 5], [5, 8], [10, 13], [17, 20], [26, 29]]))

    def test_process_dataarray(self):
        dataset = DataSet1D()
        input_name = 'y'
        output_name = 'y5'
        processed_dataset = process_dataarray(dataset, input_name, output_name, lambda x: x + 5)
        self.assertIn(output_name, processed_dataset.arrays)
        np.testing.assert_array_equal(np.array(getattr(dataset, input_name)) + 5, dataset.arrays[output_name])

    def test_process_dataarray_inplace(self):
        dataset = DataSet1D()
        input_name = 'y'
        output_name = None
        processed_dataset = process_dataarray(dataset, input_name, output_name, lambda x: x + 5)
        self.assertEqual(set(processed_dataset.arrays.keys()), {'x_set', 'y'})
        np.testing.assert_array_equal(np.array(getattr(dataset, input_name)), np.array([8., 9., 10., 11., 12.]))

    def test_average_dataset(self):
        dataset2d = DataSet2D()

        d = average_dataset(dataset2d, axis=0)
        self.assertEqual(d.z.shape, (4,))

        d = average_dataset(dataset2d, axis=1)
        self.assertEqual(d.z.shape, (6,))

    def test_dataset_dimension(self):
        dataset1d = DataSet1D()
        self.assertEqual(dataset_dimension(dataset1d), 1)
        dataset2d = DataSet2D()
        self.assertEqual(dataset_dimension(dataset2d), 2)

    def test_average_multirow_dataset(self):
        dataset2d = DataSet2D()

        averaged_dataset = average_multirow_dataset(dataset2d, 2)
        self.assertEqual(averaged_dataset.signal.shape, (3, 4))
        np.testing.assert_array_equal(averaged_dataset.signal, np.array(
            [[0.5, 1.5, 4.5, 9.5], [6.5, 7.5, 10.5, 15.5], [20.5, 21.5, 24.5, 29.5]]))

        averaged_dataset = average_multirow_dataset(dataset2d, 2, [-1, -2, -3])
        np.testing.assert_array_equal(averaged_dataset.x, [-1, -2, -3])
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(averaged_dataset.x, [-1, -2, -3, -4])

    def test_slice_dataset_copy_metadata(self):
        dataset = DataSet2D()
        dataset.metadata['a'] = 1
        dataset_sliced = slice_dataset(dataset, [2, 5])
        self.assertDictEqual(dataset_sliced.metadata, {})

        dataset_sliced = slice_dataset(dataset, [2, 5], copy_metadata=True)
        self.assertDictEqual(dataset_sliced.metadata, dataset.metadata)

    def test_resample_dataset(self):
        dataset1d = DataSet1D()
        dataset2d = DataSet2D()

        with self.assertRaises(ValueError):
            d = resample_dataset(dataset1d, (0, 1))
        d = resample_dataset(dataset1d, (4,))
        self.assertEqual(d.y.shape, (2,))

        d = resample_dataset(dataset2d, (2, 1))
        np.testing.assert_array_equal(d.z, np.array([[0, 1, 4, 9], [4, 5, 8, 13], [16, 17, 20, 25]]))

        d = average_dataset(dataset2d, axis=1)
        self.assertEqual(d.z.shape, (6,))
