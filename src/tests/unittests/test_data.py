import unittest
import time
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import qcodes
from qcodes import ManualParameter
from qcodes.data.hdf5_format import HDF5Format
from qcodes.data.gnuplot_format import GNUPlotFormat
from qcodes.data.io import DiskIO
from qcodes.tests.data_mocks import DataSet2D

import qtt.data
from qtt.data import image_transform, dataset_to_dictionary, dictionary_to_dataset,\
     compare_dataset_metadata, diffDataset, add_comment, load_dataset, determine_parameter_unit


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
        x = all_data.z
        _ = np.array(x)
        s = np.linalg.svd(x)
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

    def test_determine_parameter_unit_ok(self):
        p = ManualParameter('dummy')
        unit = determine_parameter_unit(p)
        self.assertTrue(unit == '')

    def test_determine_parameter_unit_nok(self):
        p = 'p is a string'
        unit = determine_parameter_unit(p)
        self.assertTrue(unit is None)


class TestMakeDataSet(unittest.TestCase):

    def test_makedataset1dplain_double_measurement(self):
        x = np.arange(0, 10)
        y = np.vstack((x - 1, x + 10))
        data_set = qtt.data.makeDataSet1Dplain('x', x, ['y1', 'y2'], [y[0], y[1]])

        self.assertTrue(data_set.x.shape == np.ones(len(x)).shape)
        self.assertTrue(data_set.y1.shape == np.ones(len(y[0])).shape)
        self.assertTrue(data_set.y2.shape == np.ones(len(y[1])).shape)
        # check array
        self.assertTrue(data_set.arrays['y1'].shape == np.ones(len(y[0])).shape)
        self.assertTrue(data_set.arrays['y2'].shape == np.ones(len(y[1])).shape)

    def test_makedataset1dplain_single_measurement(self):
        x = np.arange(0, 10)
        y = np.arange(1, 11)
        data_set = qtt.data.makeDataSet1Dplain('x', x, 'y', y)

        self.assertTrue(data_set.x.shape == np.ones(len(x)).shape)
        self.assertTrue(data_set.y.shape == np.ones(len(y)).shape)
        # check array
        self.assertTrue(data_set.arrays['y'].shape == np.ones(len(y)).shape)

    def test_makedataset1dplain_no_measurement(self):
        x = np.arange(0, 8)
        data_set = qtt.data.makeDataSet1Dplain('x', x, 'y')

        self.assertTrue(data_set.x.shape == np.ones(len(x)).shape)
        self.assertTrue(data_set.y.shape == np.ones(len(x)).shape)

    def test_makedataset1dplain_shape_measuredata_list_nok1(self):
        x = np.arange(0, 10)
        y = np.arange(1, 11)
        self.assertRaisesRegex(ValueError, 'The number of measurement names does not match the number of measurements',
                               qtt.data.makeDataSet1Dplain, 'x', x, ['y1', 'y2'], y)

    def test_makedataset1dplain_shape_measuredata_list_nok2(self):
        x = np.arange(0, 10)
        y = np.arange(1, 11)
        self.assertRaisesRegex(ValueError, 'The number of measurement names does not match the number of measurements',
                               qtt.data.makeDataSet1Dplain, 'x', x, 'y1', [y, y])

    def test_makedataset1dplain_type_yname_parameter(self):
        x = np.arange(0, 10)
        yname = ManualParameter('dummy')
        data_set = qtt.data.makeDataSet1Dplain('x', x, yname)
        self.assertTrue(data_set.x.shape == np.ones(len(x)).shape)
        self.assertTrue(data_set.dummy.shape == np.ones(len(x)).shape)

    def test_makedataset1dplain_type_measurement_names_nok(self):
        x = np.arange(0, 10)
        y = np.arange(1, 11)
        self.assertRaisesRegex(TypeError, 'Type of measurement names must be str or qcodes.Parameter',
                               qtt.data.makeDataSet1Dplain, 'x', x, [1, 2], [y, y])

    def test_dataset1dplain_shape_measuredata_nok(self):
        x = np.arange(0, 8)
        y = np.arange(1, 11)
        self.assertRaisesRegex(ValueError, 'Measured data must be a sequence with shape matching the setpoint arrays',
                               qtt.data.makeDataSet1Dplain, 'x', x, ['y1', 'y2'], [y, y])

    def test_dataset1dplain_shape_measuredata_second_nok(self):
        x = np.arange(0, 8)
        y1 = np.arange(1, 9)
        y2 = np.arange(1, 10)
        self.assertRaisesRegex(ValueError, 'Measured data must be a sequence with shape matching the setpoint arrays',
                               qtt.data.makeDataSet1Dplain, 'x', x, ['y1', 'y2'], [y1, y2])

    def test_makedataset1d_not_return_names(self):
        p = ManualParameter('dummy')
        x = p[0:10:1]
        yname = 'measured'
        y = np.arange(len(x)).reshape((len(x)))
        data_set = qtt.data.makeDataSet1D(x, yname, y, return_names=False)
        # check attribute
        self.assertTrue(data_set.measured.shape == np.ones(len(y)).shape)
        # check array
        self.assertTrue(data_set.arrays['measured'].shape == np.ones(len(y)).shape)

    def test_makedataset1d_type_parameter_nok1(self):
        p = ManualParameter('dummy')
        x = p[0:10:1]
        x.parameter = 4
        self.assertRaisesRegex(TypeError, 'Type of parameter.parameter must be qcodes.Parameter',
                               qtt.data.makeDataSet1D, x, [1, 2], None)

    def test_makedataset1d_type_parameter_nok2(self):
        p = 4
        self.assertRaisesRegex(TypeError, 'Type of parameter must be qcodes.SweepFixedValues',
                               qtt.data.makeDataSet1D, p, [1, 2], None)

    def test_makedataset1d_type_measurement_names_nok(self):
        p = ManualParameter('dummy')
        x = p[0:10:1]
        self.assertRaisesRegex(TypeError, 'Type of measurement names must be str or qcodes.Parameter',
                               qtt.data.makeDataSet1D, x, [1, 2], None)

    def test_makedataset1d_shape_measuredata_list_nok1(self):
        p = ManualParameter('dummy')
        x = p[0:10:1]
        yname = ['measured1', 'measured2']
        y = np.arange(len(x)).reshape((len(x)))
        self.assertRaisesRegex(ValueError, 'The number of measurement names does not match the number of measurements',
                               qtt.data.makeDataSet1D, x, yname, y, return_names=False)

    def test_makedataset1d_shape_measuredata_list_nok2(self):
        p = ManualParameter('dummy')
        x = p[0:10:1]
        yname = 'measured'
        y = [np.arange(len(x)).reshape((len(x))), np.arange(len(x)).reshape((len(x)))]
        self.assertRaisesRegex(ValueError, 'The number of measurement names does not match the number of measurements',
                               qtt.data.makeDataSet1D, x, yname, y, return_names=False)

    def test_makedataset1d_shape_measuredata_nok(self):
        p = ManualParameter('dummy')
        x = p[0:10:1]
        y = [np.arange(len(x)+1)]

        self.assertRaisesRegex(ValueError,
                               'Measured data must be a sequence with shape matching the setpoint arrays shape',
                               qtt.data.makeDataSet1D, x, 'y', y, return_names=False)

    def test_makedataset1d_shape_measuredata_second_nok(self):
        p = ManualParameter('dummy')
        x = p[0:10:1]
        y = [np.arange(len(x)), np.arange(len(x)+1)]

        self.assertRaisesRegex(ValueError,
                               'Measured data must be a sequence with shape matching the setpoint arrays shape',
                               qtt.data.makeDataSet1D, x,
                               ['y1', 'y2'], y, return_names=False)

    def test_dataset_labels_dataset2dplain(self):
        v = [0]
        h = [0]
        measurement = np.arange(len(v) * len(h)).reshape((len(v), len(h)))

        ds = qtt.data.makeDataSet2Dplain('horizontal', h, 'vertical', v, 'z', measurement,
                                         xunit='mV', yunit='Hz', zunit='A')

        self.assertEqual(qtt.data.dataset_labels(ds), 'z')
        self.assertEqual(qtt.data.dataset_labels(ds, 'x'), 'horizontal')
        self.assertEqual(qtt.data.dataset_labels(ds, 1), 'horizontal')
        self.assertEqual(qtt.data.dataset_labels(ds, 'y'), 'vertical')
        self.assertEqual(qtt.data.dataset_labels(ds, 'y', add_unit='True'), 'vertical [Hz]')

    def test_makedataset2dplain_type_measurement_names_nok(self):
        v = [0, 1, 2.]
        h = [0., 1, 2, 3, 4]
        measurement = np.arange(len(v) * len(h)).reshape((len(v), len(h)))
        z = [measurement, measurement]
        self.assertRaisesRegex(TypeError, 'Type of measurement names must be str or qcodes.Parameter',
                               qtt.data.makeDataSet2Dplain, 'horizontal', h, 'vertical', v, ['z1', 2],
                               z, xunit='mV', yunit='Hz', zunit='A')

    def test_dataset2dplain_shape_measuredata_list_ok(self):
        v = [0, 1, 2.]
        h = [0., 1, 2, 3, 4]
        measurement = np.arange(len(v) * len(h)).reshape((len(v), len(h)))
        z = [measurement, measurement]
        data_set = qtt.data.makeDataSet2Dplain('horizontal', h, 'vertical', v, ['z1', 'z2'],
                                               z, xunit='mV', yunit='Hz', zunit='A')
        self.assertTrue(data_set.z1.shape == np.ones((len(v), len(h))).shape)
        self.assertTrue(data_set.z2.shape == np.ones((len(v), len(h))).shape)
        self.assertTrue(data_set.arrays['z1'].shape == np.ones((len(v), len(h))).shape)
        self.assertTrue(data_set.arrays['z2'].shape == np.ones((len(v), len(h))).shape)

    def test_dataset2dplain_shape_measuredata_list_nok1(self):
        v = [0, 1, 2.]
        h = [0., 1, 2, 3, 4]
        measurement = np.arange(len(v) * len(h)).reshape((len(v), len(h)))
        z = [measurement, measurement, measurement]
        self.assertRaisesRegex(ValueError, 'The number of measurement names does not match the number of measurements',
                               qtt.data.makeDataSet2Dplain, 'horizontal', h, 'vertical', v, ['z1', 'z2'],
                               z, xunit='mV', yunit='Hz', zunit='A')

    def test_dataset2dplain_shape_measuredata_list_nok2(self):
        v = [0, 1, 2.]
        h = [0., 1, 2, 3, 4]
        measurement = np.arange(len(v) * len(h)).reshape((len(v), len(h)))
        z = [measurement, measurement]
        self.assertRaisesRegex(ValueError, 'The number of measurement names does not match the number of measurements',
                               qtt.data.makeDataSet2Dplain, 'horizontal', h, 'vertical', v, ['z1', 'z2', 'z3'],
                               z, xunit='mV', yunit='Hz', zunit='A')

    def test_dataset2dplain_shape_measuredata_ok(self):
        v = [0, 1, 2.]
        h = [0., 1, 2, 3, 4]
        z = np.arange(len(v) * len(h)).reshape((len(v), len(h)))
        data_set = qtt.data.makeDataSet2Dplain('horizontal', h, 'vertical', v, 'z',
                                               z, xunit='mV', yunit='Hz', zunit='A')
        self.assertTrue(data_set.z.shape == np.ones((len(v), len(h))).shape)
        self.assertTrue(data_set.arrays['z'].shape == np.ones((len(v), len(h))).shape)

    def test_dataset2dplain_shape_measuredata_nok(self):
        self.assertRaisesRegex(ValueError, 'Measured data must be a sequence with shape matching the setpoint arrays',
                               qtt.data.makeDataSet2Dplain, 'horizontal', [0., 1, 2, 3], 'vertical', [0, 1, 2.], 'z',
                               np.arange(3 * 5).reshape((3, 5)), xunit='mV', yunit='Hz', zunit='A')

    def test_dataset2dplain_shape_measuredata_second_nok(self):
        self.assertRaisesRegex(ValueError, 'Measured data must be a sequence with shape matching the setpoint arrays',
                               qtt.data.makeDataSet2Dplain, 'horizontal', [0., 1, 2, 3], 'vertical', [0, 1, 2.],
                               ['z1', 'z2'], [np.arange(3 * 4).reshape((3, 4)), np.arange(3 * 5).reshape((3, 5))],
                               xunit='mV', yunit='Hz', zunit='A')

    def test_dataset2dplain_no_measuredata(self):
        v = [0, 1, 2.]
        h = [0., 1, 2, 3, 4]
        data_set = qtt.data.makeDataSet2Dplain('horizontal', h, 'vertical', v,
                                               xunit='mV', yunit='Hz', zunit='A')
        self.assertTrue(data_set.measured.shape == np.ones((len(v), len(h))).shape)
        self.assertTrue(data_set.arrays['measured'].shape == np.ones((len(v), len(h))).shape)

    @staticmethod
    def test_makedataset2d_diffdataset():
        p1 = ManualParameter('dummy1')
        p2 = ManualParameter('dummy2')
        ds = qtt.data.makeDataSet2D(p1[0:10:1], p2[0:4:1], ['m1', 'm2'])
        _ = diffDataset(ds)

    def test_makedataset2d_not_return_names(self):
        p1 = ManualParameter('dummy1')
        p2 = ManualParameter('dummy2')
        x = p1[0:10:1]
        y = p2[0:4:1]
        measure_names = 'measured'
        # preset_data = np.ones((len(x), len(y)))
        data_set = qtt.data.makeDataSet2D(x, y, measure_names, return_names=False)
        # check attribute
        self.assertTrue(data_set.measured.shape == np.ones((len(x), len(y))).shape)
        # check array
        self.assertTrue(data_set.arrays['measured'].shape == np.ones((len(x), len(y))).shape)

    def test_makedataset2d_return_names(self):
        p1 = ManualParameter('dummy1')
        p2 = ManualParameter('dummy2')
        x = p1[0:10:1]
        y = p2[0:4:1]
        measure_names = 'measured'
        data_set, tuple_names = qtt.data.makeDataSet2D(x, y, measure_names, return_names=True)
        self.assertTrue(data_set.measured.shape == np.ones((len(x), len(y))).shape)
        self.assertTrue(data_set.arrays['measured'].shape == np.ones((len(x), len(y))).shape)
        self.assertTrue(tuple_names[0][0] == 'dummy1')
        self.assertTrue(tuple_names[0][1] == 'dummy2')
        self.assertTrue(tuple_names[1][0] == 'measured')

    def test_makedataset2d_preset_data_is_arry(self):
        p1 = ManualParameter('dummy1')
        p2 = ManualParameter('dummy2')
        x = p1[0:10:1]
        y = p2[0:4:1]
        m1 = ManualParameter('measurement1')
        measure_names = [m1]
        preset_data = np.ones((len(x), len(y)))
        data_set = qtt.data.makeDataSet2D(x, y, measure_names, preset_data=preset_data, return_names=False)
        self.assertTrue(data_set.measurement1.shape == np.ones((len(x), len(y))).shape)
        self.assertTrue(data_set.arrays['measurement1'].shape == np.ones((len(x), len(y))).shape)

    def test_makedataset2d_shape_measure_names_parameters(self):
        p1 = ManualParameter('dummy1')
        p2 = ManualParameter('dummy2')
        x = p1[0:10:1]
        y = p2[0:4:1]
        m1 = ManualParameter('measurement1')
        m2 = ManualParameter('measurement2')
        measure_names = [m1, m2]
        preset_data = [np.ones((len(x), len(y))), np.ones((len(x), len(y)))]
        data_set = qtt.data.makeDataSet2D(x, y, measure_names, preset_data=preset_data, return_names=False)
        self.assertTrue(data_set.measurement1.shape == np.ones((len(x), len(y))).shape)
        self.assertTrue(data_set.arrays['measurement2'].shape == np.ones((len(x), len(y))).shape)

    def test_makedataset2d_type_parameter_nok1(self):
        p1 = ManualParameter('dummy1')
        p2 = ManualParameter('dummy2')
        x = p1[0:10:1]
        y = p2[0:4:1]
        x.parameter = 4
        measure_names = 'measured'
        self.assertRaisesRegex(TypeError, 'Type of parameter.parameter must be qcodes.Parameter',
                               qtt.data.makeDataSet2D, x, y, measure_names, return_names=False)

    def test_makedataset2d_type_parameter_nok2(self):
        p1 = ManualParameter('dummy1')
        x = p1[0:10:1]
        y = 'wrong type'
        measure_names = 'measured'
        self.assertRaisesRegex(TypeError, 'Type of parameter must be qcodes.SweepFixedValues',
                               qtt.data.makeDataSet2D, x, y, measure_names, return_names=False)

    def test_makedataset2d_type_measurement_names_nok(self):
        p1 = ManualParameter('dummy1')
        p2 = ManualParameter('dummy2')
        x = p1[0:10:1]
        y = p2[0:4:1]
        measure_names = [2]
        self.assertRaisesRegex(TypeError, 'Type of measurement names must be str or qcodes.Parameter',
                               qtt.data.makeDataSet2D, x, y, measure_names, return_names=False)

    def test_makedataset2d_shape_measuredata_list_nok1(self):
        p1 = ManualParameter('dummy1')
        p2 = ManualParameter('dummy2')
        x = p1[0:10:1]
        y = p2[0:4:1]
        measure_names = ['measured1', 'measured2']
        preset_data = [np.ones((len(x), len(y))).shape]
        self.assertRaisesRegex(ValueError, 'The number of measurement names does not match the number of measurements',
                               qtt.data.makeDataSet2D, x, y, measure_names, preset_data=preset_data, return_names=False)

    def test_makedataset2d_shape_measuredata_list_nok2(self):
        p1 = ManualParameter('dummy1')
        p2 = ManualParameter('dummy2')
        x = p1[0:10:1]
        y = p2[0:4:1]
        measure_names = ['measured1']
        preset_data = [np.ones((len(x), len(y))).shape, np.ones((len(x), len(y))).shape]
        self.assertRaisesRegex(ValueError, 'The number of measurement names does not match the number of measurements',
                               qtt.data.makeDataSet2D, x, y, measure_names, preset_data=preset_data, return_names=False)

    def test_makedataset2d_shape_measuredata_nok(self):
        p1 = ManualParameter('dummy1')
        p2 = ManualParameter('dummy2')
        x = p1[0:10:1]
        y = p2[0:4:1]
        measure_names = ['measured1', 'measured2']
        preset_data = [np.ones((len(x) + 1, len(y))), np.ones((len(x), len(y)))]
        self.assertRaisesRegex(ValueError,
                               'Measured data must be a sequence with shape matching the setpoint arrays shape',
                               qtt.data.makeDataSet2D, x, y, measure_names, preset_data=preset_data, return_names=False)

    def test_makedataset2d_shape_measuredata_second_nok(self):
        p1 = ManualParameter('dummy1')
        p2 = ManualParameter('dummy2')
        x = p1[0:10:1]
        y = p2[0:4:1]
        measure_names = ['measured1', 'measured2']
        preset_data = [np.ones((len(x), len(y))), np.ones((len(x) + 1, len(y)))]
        self.assertRaisesRegex(ValueError,
                               'Measured data must be a sequence with shape matching the setpoint arrays shape',
                               qtt.data.makeDataSet2D, x, y, measure_names, preset_data=preset_data, return_names=False)
