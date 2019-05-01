import unittest
import numpy as np
import matplotlib.pyplot as plt

import qcodes
import qcodes.tests.data_mocks

import qtt.data


# %%

class TestPlotting(unittest.TestCase):

    def test_plot_dataset_1d(self):
        dataset = qcodes.tests.data_mocks.DataSet1D()
        qtt.data.plot_dataset(dataset, fig=1)
        self.assertTrue(plt.fignum_exists(1))
        plt.close(1)

    def test_plot_dataset_2d(self):
        dataset = qtt.data.makeDataSet2Dplain('horizontal', [0., 1, 2, 3], 'vertical', [0, 1, 2.], 'z',
                                              np.arange(3 * 4).reshape((3, 4)), xunit='mV', yunit='Hz', zunit='A')
        qtt.data.plot_dataset(dataset, fig=2)
        self.assertTrue(plt.fignum_exists(2))
        plt.close(1)


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
        resolution = tr.scan_resolution()
        self.assertEqual(resolution, 1)

    def test_dataset1Dmetadata(self):
        dataset = qcodes.tests.data_mocks.DataSet1D(name='test1d')
        _, _, _, _, arrayname = qtt.data.dataset1Dmetadata(dataset)
        self.assertEqual(arrayname, 'y')

    def test_datasetCentre(self):
        dataset = qcodes.tests.data_mocks.DataSet2D()
        cc = qtt.data.datasetCentre(dataset)
        assert (cc[0] == 1.5)

    def test_dataset_labels(self):
        dataset = qcodes.tests.data_mocks.DataSet2D()
        zz = qtt.data.dataset_labels(dataset)
        self.assertEqual(zz, 'Z')
        zz = qtt.data.dataset_labels(dataset, add_unit=True)
        self.assertEqual(zz, 'Z [None]')
