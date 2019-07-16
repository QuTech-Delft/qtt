import unittest
import qcodes.tests.data_mocks

from qtt.dataset_processing import slice_dataset


class TestDataProcessing(unittest.TestCase):

    def test_slice_dataset(self):
        dataset = qcodes.tests.data_mocks.DataSet1D()
        dataset_sliced = slice_dataset(dataset, [2,5] )
        self.assertEqual(dataset_sliced.default_parameter_array().shape, (3,))

        dataset = qcodes.tests.data_mocks.DataSet2D()
        dataset_sliced = slice_dataset(dataset, [1,5], axis=0)

