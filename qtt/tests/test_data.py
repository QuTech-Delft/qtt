import unittest

import qcodes
import qcodes.tests.data_mocks

import qtt.data

#%%


class TestDataSetHelpers(unittest.TestCase):

    def test_dataset_labels(self):
        ds = qtt.data.makeDataSet2Dplain('horizontal', [0], 'vertical', [0], 'z', [
                                         0], xunit='mV', yunit='Hz', zunit='A')

        self.assertEqual(qtt.data.dataset_labels(ds), 'z')
        self.assertEqual(qtt.data.dataset_labels(ds, 'x'), 'horizontal')
        self.assertEqual(qtt.data.dataset_labels(ds, 1), 'horizontal')
        self.assertEqual(qtt.data.dataset_labels(ds, 'y'), 'vertical')

        self.assertEqual(qtt.data.dataset_labels(ds, 'y', add_unit='True'), 'vertical [Hz]')


class TestData(unittest.TestCase):

    def test_transform(self):
        dataset = qcodes.tests.data_mocks.DataSet2D()
        tr = qtt.data.image_transform(dataset, arrayname='z')
        istep = tr.istep()
        self.assertEqual(istep, 1)


if __name__ == '__main__':
    unittest.main()

    t1 = TestData()
    t1.test_transform()
