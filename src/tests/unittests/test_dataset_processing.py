import unittest
import qcodes.tests.data_mocks
import numpy as np

from qtt.dataset_processing import slice_dataset, process_dataarray, average_dataset


class TestDataProcessing(unittest.TestCase):

    def test_slice_dataset(self):
        dataset = qcodes.tests.data_mocks.DataSet1D()
        dataset_sliced = slice_dataset(dataset, [2, 5])
        self.assertEqual(dataset_sliced.default_parameter_array().shape, (3,))

        dataset = qcodes.tests.data_mocks.DataSet2D()
        dataset_sliced = slice_dataset(dataset, [1, 5], axis=0)

    def test_process_dataarray(self):
        dataset = qcodes.tests.data_mocks.DataSet1D()
        input_name = 'y'
        output_name = 'y5'
        processed_dataset = process_dataarray(dataset, input_name, output_name, lambda x: x + 5)
        self.assertIn(output_name, processed_dataset.arrays)
        np.testing.assert_array_equal(np.array(getattr(dataset, input_name)) + 5, dataset.arrays[output_name])

    def test_average_dataset(self):
        dataset2d = qcodes.tests.data_mocks.DataSet2D()

        d = average_dataset(dataset2d, axis=0)
        self.assertEqual(d.z.shape, (4, ))

        d = average_dataset(dataset2d, axis=1)
        self.assertEqual(d.z.shape, (6, ))
