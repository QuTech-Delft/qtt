import unittest
import time
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import qcodes
import qcodes.tests.data_mocks
from qcodes.tests.data_mocks import DataSet2D
import qtt.data
from qtt.data import image_transform, dataset_to_dictionary, dictionary_to_dataset, makeDataSet2D,\
    compare_dataset_metadata, diffDataset, makeDataSet1Dplain, add_comment, load_dataset


class TestPlotting(unittest.TestCase):

    def test_plot_dataset_1d(self, fig=None):
        dataset = qcodes.tests.data_mocks.DataSet1D()
        if fig is not None:
            qtt.data.plot_dataset(dataset, fig=fig)
            self.assertTrue(plt.fignum_exists(fig))
            plt.close(fig)

    def test_plot_dataset_2d(self, fig=None):
        dataset = qtt.data.makeDataSet2Dplain('horizontal', [0., 1, 2, 3], 'vertical', [0, 1, 2.], 'z',
                                              np.arange(3 * 4).reshape((3, 4)), xunit='mV', yunit='Hz', zunit='A')
        if fig is not None:
            qtt.data.plot_dataset(dataset, fig=fig)
            self.assertTrue(plt.fignum_exists(fig))
            plt.close(fig)


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
        self.assertEqual(cc[0], 1.5)

    def test_dataset_labels_dataset_2d(self):
        dataset = qcodes.tests.data_mocks.DataSet2D()
        zz = qtt.data.dataset_labels(dataset)
        self.assertEqual(zz, 'Z')
        zz = qtt.data.dataset_labels(dataset, add_unit=True)
        self.assertEqual(zz, 'Z [None]')

    def test_dataset_labels_dataset_2d_plain(self):
        ds = qtt.data.makeDataSet2Dplain('horizontal', [0], 'vertical', [0], 'z', [
            0], xunit='mV', yunit='Hz', zunit='A')

        self.assertEqual(qtt.data.dataset_labels(ds), 'z')
        self.assertEqual(qtt.data.dataset_labels(ds, 'x'), 'horizontal')
        self.assertEqual(qtt.data.dataset_labels(ds, 1), 'horizontal')
        self.assertEqual(qtt.data.dataset_labels(ds, 'y'), 'vertical')

        self.assertEqual(qtt.data.dataset_labels(ds, 'y', add_unit='True'), 'vertical [Hz]')

    def test_dataset_to_dictionary(self):

        input_dataset = qcodes.tests.data_mocks.DataSet2D()

        data_dictionary = dataset_to_dictionary(input_dataset, include_data=False, include_metadata=False)
        self.assertIsNone(data_dictionary['metadata'])

        data_dictionary = dataset_to_dictionary(input_dataset, include_data=True, include_metadata=True)
        self.assertTrue('metadata' in data_dictionary)

        converted_dataset = dictionary_to_dataset(data_dictionary)
        self.assertEqual(converted_dataset.default_parameter_name(), input_dataset.default_parameter_name())

    @staticmethod
    def test_makeDataSet2D():
        from qcodes import ManualParameter
        p = ManualParameter('dummy')
        p2 = ManualParameter('dummy2')
        ds = makeDataSet2D(p[0:10:1], p2[0:4:1], ['m1', 'm2'])
        _ = diffDataset(ds)

    @staticmethod
    def test_makeDataSet1Dplain():
        x = np.arange(0, 10)
        y = np.vstack((x - 1, x + 10))
        _ = makeDataSet1Dplain('x', x, ['y1', 'y2'], y)

    @staticmethod
    def test_numpy_on_dataset(verbose=0):
        all_data = qcodes.tests.data_mocks.DataSet2D()
        X = all_data.z
        _ = np.array(X)
        s = np.linalg.svd(X)
        if verbose:
            print(s)

    @staticmethod
    def test_compare():
        ds = qcodes.tests.data_mocks.DataSet2D()
        compare_dataset_metadata(ds, ds, verbose=0)

    @staticmethod
    def test_image_transform(verbose=0):
        ds = DataSet2D()
        tr = image_transform(ds)
        im = tr.image()
        if verbose:
            print('transform: im.shape %s' % (str(im.shape),))
        tr = image_transform(ds, unitsperpixel=[None, 2])
        im = tr.image()
        if verbose:
            print('transform: im.shape %s' % (str(im.shape),))

    def test_add_comment(self):
        ds0 = qcodes.tests.data_mocks.DataSet2D()
        ds = qcodes.tests.data_mocks.DataSet2D()
        try:
            add_comment('hello world')
        except NotImplementedError:
            ds.metadata['comment'] = 'hello world'

        add_comment('hello world 0', ds0)
        self.assertTrue(ds.metadata['comment'] == 'hello world')
        self.assertTrue(ds0.metadata['comment'] == 'hello world 0')

    @staticmethod
    def test_load_dataset(verbose=0):
        h = qcodes.data.hdf5_format.HDF5Format()
        g = qcodes.data.gnuplot_format.GNUPlotFormat()

        io = qcodes.data.io.DiskIO(tempfile.mkdtemp())
        dd = []
        name = qcodes.DataSet.location_provider.base_record['name']
        for jj, fmt in enumerate([g, h]):
            ds = qcodes.tests.data_mocks.DataSet2D(name='format%d' % jj)
            ds.formatter = fmt
            ds.io = io
            ds.add_metadata({'1': 1, '2': [2, 'x'], 'np': np.array([1, 2.])})
            ds.write(write_metadata=True)
            dd.append(ds.location)
            time.sleep(.1)
        qcodes.DataSet.location_provider.base_record['name'] = name

        for _, location in enumerate(dd):
            if verbose:
                print('load %s' % location)
            r = load_dataset(location, io=io)
            if verbose:
                print(r)
