import io
import logging
import tempfile
import time
import sys
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import qcodes
from qcodes import ManualParameter
from qcodes.data.data_set import DataSet
from qcodes.tests.legacy.data_mocks import DataSet1D, DataSet2D

import qtt.data
from qtt.data import image_transform, dataset_to_dictionary, dictionary_to_dataset,\
    compare_dataset_metadata, diffDataset, add_comment, load_dataset, determine_parameter_unit
from qtt.data import logger


class TestPlotting(unittest.TestCase):

    def test_plot_dataset_1d(self, fig=None):
        dataset = DataSet1D()
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

class TestExampleDatasets(unittest.TestCase):

    def test_json_format(self):
        dataset=qtt.data.load_example_dataset('elzerman_detuning_scan.json')
        self.assertEqual(dataset.default_parameter_name(), 'signal')
        self.assertEqual(dataset.default_parameter_array().shape, (240, 1367))
        np.testing.assert_almost_equal(dataset.time[0, 0::500], np.array([0., 0.000512, 0.001024]))

    def test_qcodes_hdf5_format(self):
        dataset=qtt.data.load_example_dataset('polarization_line')
        self.assertEqual(dataset.default_parameter_name(), 'signal')
        np.testing.assert_almost_equal(dataset.delta[0:3], np.array([-100.,  -99.79979706,  -99.59960175]))


class TestData(unittest.TestCase):

    def test_transform(self):
        dataset = DataSet2D()
        tr = qtt.data.image_transform(dataset, arrayname='z')
        resolution = tr.scan_resolution()
        self.assertEqual(resolution, 1)

    def test_dataset1Dmetadata(self):
        dataset = DataSet1D(name='test1d')
        _, _, _, _, arrayname = qtt.data.dataset1Dmetadata(dataset)
        self.assertEqual(arrayname, 'y')

    def test_datasetCentre(self):
        dataset = DataSet2D()
        cc = qtt.data.datasetCentre(dataset)
        self.assertEqual(cc[0], 1.5)

    def test_dataset_labels_dataset_1d(self):
        dataset = qtt.data.makeDataSet1Dplain('x', [0, 1], 'y', [2, 3])
        independent_label = qtt.data.dataset_labels(dataset, 'x')
        self.assertEqual(independent_label, 'x')
        independent_label = qtt.data.dataset_labels(dataset, 0)
        self.assertEqual(independent_label, 'x')
        dependent_label = qtt.data.dataset_labels(dataset)
        self.assertEqual(dependent_label, 'y')
        dependent_label = qtt.data.dataset_labels(dataset, 'z')
        self.assertEqual(dependent_label, 'y')

    def test_dataset_labels_dataset_2d(self):
        dataset = DataSet2D()
        zz = qtt.data.dataset_labels(dataset)
        self.assertEqual(zz, 'Z')
        zz = qtt.data.dataset_labels(dataset, add_unit=True)
        self.assertEqual(zz, 'Z [None]')

    def test_dataset_labels_dataset2dplain(self):
        v = [0]
        h = [0]
        measurement = np.arange(len(v) * len(h)).reshape((len(v), len(h)))

        ds = qtt.data.makeDataSet2Dplain('horizontal', h, 'vertical', v, 'z', measurement,
                                         xunit='mV', yunit='Hz', zunit='A')

        # check labels
        self.assertEqual(qtt.data.dataset_labels(ds), 'z')
        self.assertEqual(qtt.data.dataset_labels(ds, 'x'), 'horizontal')
        self.assertEqual(qtt.data.dataset_labels(ds, 1), 'vertical')
        self.assertEqual(qtt.data.dataset_labels(ds, 'y'), 'vertical')
        self.assertEqual(qtt.data.dataset_labels(ds, 'y', add_unit=True), 'vertical [Hz]')

    def test_dataset_to_dictionary(self):

        input_dataset = DataSet2D()

        data_dictionary = dataset_to_dictionary(input_dataset, include_data=False, include_metadata=False)
        self.assertIsNone(data_dictionary['metadata'])

        data_dictionary = dataset_to_dictionary(input_dataset, include_data=True, include_metadata=True)
        self.assertTrue('metadata' in data_dictionary)

        converted_dataset = dictionary_to_dataset(data_dictionary)
        self.assertEqual(converted_dataset.default_parameter_name(), input_dataset.default_parameter_name())

    @staticmethod
    def test_numpy_on_dataset(verbose=0):
        all_data = DataSet2D()
        x = all_data.z
        _ = np.array(x)
        s = np.linalg.svd(x)
        if verbose:
            print(s)

    @staticmethod
    def test_compare():
        ds = DataSet2D()
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
        ds0 = DataSet2D()
        ds = DataSet2D()
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

        disk_io = qcodes.data.io.DiskIO(tempfile.mkdtemp())
        dd = []
        name = DataSet.location_provider.base_record['name']
        for jj, fmt in enumerate([g, h]):
            ds = DataSet2D(name='format%d' % jj)
            ds.formatter = fmt
            ds.io = disk_io
            ds.add_metadata({'1': 1, '2': [2, 'x'], 'np': np.array([1, 2.])})
            ds.write(write_metadata=True)
            dd.append(ds.location)
            time.sleep(.1)
        DataSet.location_provider.base_record['name'] = name

        for _, location in enumerate(dd):
            if verbose:
                print('load %s' % location)
            r = load_dataset(location, io=disk_io)
            if verbose:
                print(r)

    def test_determine_parameter_unit_ok(self):
        dummy_parameter = ManualParameter('dummy')
        unit = determine_parameter_unit(dummy_parameter)
        self.assertTrue(unit == '')

    def test_determine_parameter_unit_nok(self):
        not_a_parameter = 'This is a string'
        unit = determine_parameter_unit(not_a_parameter)
        self.assertTrue(unit is None)


class TestMakeDataSet(unittest.TestCase):

    def setUp(self):
        logger.propagate = False

    def tearDown(self):
        logger.propagate = True

    def test_makedataset1dplain_double_measurement(self):
        x = np.arange(0, 10)
        y = np.vstack((x - 1, x + 10))
        data_set = qtt.data.makeDataSet1Dplain('x', x, ['y1', 'y2'], [y[0], y[1]])

        # check attribute
        self.assertTrue(np.array_equal(data_set.x, x))
        self.assertTrue(np.array_equal(data_set.y1, y[0]))
        self.assertTrue(np.array_equal(data_set.y2, y[1]))
        # check array
        self.assertTrue(np.array_equal(data_set.arrays['y1'], y[0]))
        self.assertTrue(np.array_equal(data_set.arrays['y2'], y[1]))

    def test_makedataset1dplain_single_measurement(self):
        x = np.arange(0, 10)
        y = np.arange(1, 11)
        data_set = qtt.data.makeDataSet1Dplain('x', x, 'y', y)

        # check attribute
        self.assertTrue(np.array_equal(data_set.x, x))
        self.assertTrue(np.array_equal(data_set.y, y))
        # check array
        self.assertTrue(np.array_equal(data_set.arrays['y'], y))

    def test_makedataset1dplain_no_measurement(self):
        x = np.arange(0, 8)
        data_set = qtt.data.makeDataSet1Dplain('x', x, 'y')

        # check attribute
        self.assertTrue(np.array_equal(data_set.x, x))
        self.assertTrue(data_set.y.shape == np.ones(len(x)).shape)

    def test_makedataset1dplain_shape_measuredata_list_nok1(self):
        x = np.arange(0, 10)
        y = np.arange(1, 11)
        self.assertRaisesRegex(ValueError, 'The number of measurement names 2 does not match the number of measurements 10',
                               qtt.data.makeDataSet1Dplain, 'x', x, ['y1', 'y2'], y)

    def test_makedataset1dplain_shape_measuredata_list_nok2(self):
        x = np.arange(0, 10)
        y = np.arange(1, 11)
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            stream_handler = logging.StreamHandler(sys.stdout)
            logger.addHandler(stream_handler)

            data_set = qtt.data.makeDataSet1Dplain('x', x, 'y1', [y, y])
            self.assertTrue(np.array_equal(data_set.x, x))
            self.assertTrue(np.array_equal(data_set.y1, [y, y]))

            # Verify warning
            print_string = mock_stdout.getvalue()
            self.assertRegex(print_string, 'Shape of measured data .* does not match setpoint shape .*' )
            logger.removeHandler(stream_handler)

    def test_makedataset1dplain_type_yname_parameter(self):
        x = np.arange(0, 10)
        yname = ManualParameter('dummy')
        data_set = qtt.data.makeDataSet1Dplain('x', x, yname)

        # check attribute
        self.assertTrue(np.array_equal(data_set.x, x))
        self.assertTrue(data_set.dummy.shape == np.ones(len(x)).shape)
        # check array
        self.assertTrue(data_set.arrays['dummy'].shape == np.ones(len(x)).shape)

    def test_makedataset1dplain_type_measurement_names_nok(self):
        x = np.arange(0, 10)
        y = np.arange(1, 11)
        self.assertRaisesRegex(TypeError, 'Type of measurement names must be str or qcodes.Parameter',
                               qtt.data.makeDataSet1Dplain, 'x', x, [1, 2], [y, y])

    def test_dataset1dplain_shape_measuredata_nok(self):
        x = np.arange(0, 8)
        y = np.arange(1, 11)
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            stream_handler = logging.StreamHandler(sys.stdout)
            logger.addHandler(stream_handler)

            data_set = qtt.data.makeDataSet1Dplain('x', x, ['y1', 'y2'], [y, y])
            self.assertTrue(np.array_equal(data_set.x, x))
            self.assertTrue(np.array_equal(data_set.y1, y))
            self.assertTrue(np.array_equal(data_set.y2, y))

            # Verify warning
            print_string = mock_stdout.getvalue()
            self.assertRegex(print_string, 'Shape of measured data .* does not match setpoint shape .*' )
            logger.removeHandler(stream_handler)

    def test_dataset1dplain_shape_measuredata_second_nok(self):
        x = np.arange(0, 8)
        y1 = np.arange(1, 9)
        y2 = np.arange(1, 10)
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            stream_handler = logging.StreamHandler(sys.stdout)
            logger.addHandler(stream_handler)

            data_set = qtt.data.makeDataSet1Dplain('x', x, ['y1', 'y2'], [y1, y2])
            self.assertTrue(np.array_equal(data_set.x, x))
            self.assertTrue(np.array_equal(data_set.y1, y1))
            self.assertTrue(np.array_equal(data_set.y2, y2))

            # Verify warning
            print_string = mock_stdout.getvalue()
            self.assertRegex(print_string, 'Shape of measured data .* does not match setpoint shape .*' )
            logger.removeHandler(stream_handler)

    def test_dataset1dplain_right_shape(self):
        x = [1, 2, 3]
        y = np.array([1, 2, 3]).reshape((3,))
        data_set = qtt.data.makeDataSet1Dplain('x', x, 'y', y)
        self.assertTrue(np.array_equal(data_set.x, x))
        self.assertTrue(np.array_equal(data_set.y, y))

    def test_dataset1dplain_other_shape(self):
        x = [1, 2, 3]
        y = np.array([1, 2, 3]).reshape((3, 1))
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            stream_handler = logging.StreamHandler(sys.stdout)
            logger.addHandler(stream_handler)

            data_set = qtt.data.makeDataSet1Dplain('x', x, 'y', y)
            self.assertTrue(np.array_equal(data_set.x, x))
            self.assertTrue(np.array_equal(data_set.y, y))

            # Verify warning
            print_string = mock_stdout.getvalue()
            self.assertRegex(print_string, 'Shape of measured data .* does not match setpoint shape .*' )
            logger.removeHandler(stream_handler)

    def test_dataset1dplain_no_data(self):
        x = [1, 2, 3]
        y = None
        data_set = qtt.data.makeDataSet1Dplain('x', x, 'y', y)
        self.assertTrue(np.array_equal(data_set.x, x))

    def test_dataset1dplain_split_2D_input(self):
        x = [1, 2, 3]
        y = np.random.rand(2, 3)
        data_set = qtt.data.makeDataSet1Dplain('x', x, ['y1', 'y2'], y)
        self.assertTrue(np.array_equal(data_set.y1, y[0]))
        self.assertTrue(np.array_equal(data_set.y2, y[1]))
        self.assertTrue(np.array_equal(data_set.x, x))

    def test_makedataset1d_not_return_names(self):
        dummy_parameter = ManualParameter('dummy')
        x = dummy_parameter[0:10:1]
        yname = 'measured'
        y = np.arange(len(x)).reshape((len(x)))
        data_set = qtt.data.makeDataSet1D(x, yname, y, return_names=False)
        # check attribute
        self.assertTrue(np.array_equal(data_set.measured, y))
        # check array
        self.assertTrue(np.array_equal(data_set.arrays['measured'], y))

    def test_makedataset1d_type_parameter_nok1(self):
        dummy_parameter = ManualParameter('dummy')
        x = dummy_parameter[0:10:1]
        x.parameter = 4
        self.assertRaisesRegex(TypeError, 'Type of parameter.parameter must be qcodes.Parameter',
                               qtt.data.makeDataSet1D, x, [1, 2], None)

    def test_makedataset1d_type_parameter_nok2(self):
        not_of_type_parameter = 4
        self.assertRaisesRegex(TypeError, 'Type of parameter must be qcodes.SweepFixedValues',
                               qtt.data.makeDataSet1D, not_of_type_parameter, [1, 2], None)

    def test_makedataset1d_type_measurement_names_nok(self):
        dummy_parameter = ManualParameter('dummy')
        x = dummy_parameter[0:10:1]
        self.assertRaisesRegex(TypeError, 'Type of measurement names must be str or qcodes.Parameter',
                               qtt.data.makeDataSet1D, x, [1, 2], None)

    def test_makedataset1d_shape_measuredata_list_nok1(self):
        dummy_parameter = ManualParameter('dummy')
        x = dummy_parameter[0:10:1]
        yname = ['measured1', 'measured2']
        y = np.arange(len(x)).reshape((len(x)))
        self.assertRaisesRegex(ValueError, 'The number of measurement names 2 does not match the number of measurements 10',
                               qtt.data.makeDataSet1D, x, yname, y, return_names=False)

    def test_makedataset1d_shape_measuredata_list_nok2(self):
        dummy_parameter = ManualParameter('dummy')
        x = dummy_parameter[0:10:1]
        yname = 'measured'
        y = [np.arange(len(x)).reshape((len(x))), np.arange(len(x)).reshape((len(x)))]
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            stream_handler = logging.StreamHandler(sys.stdout)
            logger.addHandler(stream_handler)

            data_set = qtt.data.makeDataSet1D(x, yname, y, return_names=False)
            self.assertTrue(np.array_equal(data_set.dummy, x))
            self.assertTrue(np.array_equal(data_set.measured, y))

            # Verify warning
            print_string = mock_stdout.getvalue()
            self.assertRegex(print_string, 'Shape of measured data .* does not match setpoint shape .*' )
            logger.removeHandler(stream_handler)

    def test_makedataset1d_shape_measuredata_nok(self):
        dummy_parameter = ManualParameter('dummy')
        x = dummy_parameter[0:10:1]
        y = [np.arange(len(x)+1)]
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            stream_handler = logging.StreamHandler(sys.stdout)
            logger.addHandler(stream_handler)

            data_set = qtt.data.makeDataSet1D(x, 'y', y, return_names=False)
            self.assertTrue(np.array_equal(data_set.dummy, x))
            self.assertTrue(np.array_equal(data_set.y, y))

            # Verify warning
            print_string = mock_stdout.getvalue()
            self.assertRegex(print_string, 'Shape of measured data .* does not match setpoint shape .*' )
            logger.removeHandler(stream_handler)

    def test_makedataset1d_shape_second_item_measuredata_nok(self):
        dummy_parameter = ManualParameter('dummy')
        x = dummy_parameter[0:10:1]
        y = [np.arange(len(x)), np.arange(len(x)+1)]
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            stream_handler = logging.StreamHandler(sys.stdout)
            logger.addHandler(stream_handler)

            data_set = qtt.data.makeDataSet1D(x, ['y1', 'y2'], y, return_names=False)
            self.assertTrue(np.array_equal(data_set.dummy, x))
            self.assertTrue(np.array_equal(data_set.y1, y[0]))
            self.assertTrue(np.array_equal(data_set.y2, y[1]))

            # Verify warning
            print_string = mock_stdout.getvalue()
            self.assertRegex(print_string, 'Shape of measured data .* does not match setpoint shape .*' )
            logger.removeHandler(stream_handler)

    def test_makedataset1d_no_data(self):
        dummy_parameter = ManualParameter('dummy')
        x = dummy_parameter[0:10:1]
        y = None
        data_set = qtt.data.makeDataSet1D(x, ['y1', 'y2'], y, return_names=False)
        self.assertTrue(np.array_equal(data_set.dummy, x))
        self.assertTrue(data_set.y1.shape == (10,))
        self.assertTrue(data_set.y2.shape == (10,))

    def test_makedataset1d_return_names(self):
        dummy_parameter = ManualParameter('dummy')
        x = dummy_parameter[0:10:1]
        y = None
        data_set, tuple_names = qtt.data.makeDataSet1D(x, ['y1', 'y2'], y, return_names=True)
        self.assertTrue(np.array_equal(data_set.dummy, x))
        # check return names
        self.assertTrue(tuple_names[0] == 'dummy')
        self.assertTrue(tuple_names[1][0] == 'y1')
        self.assertTrue(tuple_names[1][1] == 'y2')

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

        # check attribute
        self.assertTrue(np.array_equal(data_set.z1, z[0]))
        self.assertTrue(np.array_equal(data_set.z2, z[1]))
        # check array
        self.assertTrue(np.array_equal(data_set.arrays['z1'], z[0]))
        self.assertTrue(np.array_equal(data_set.arrays['z2'], z[1]))

    def test_dataset2dplain_shape_measuredata_list_nok1(self):
        v = [0, 1, 2.]
        h = [0., 1, 2, 3, 4]
        measurement = np.arange(len(v) * len(h)).reshape((len(v), len(h)))
        z = [measurement, measurement, measurement]
        self.assertRaisesRegex(ValueError, 'The number of measurement names 2 does not match the number of measurements 3',
                               qtt.data.makeDataSet2Dplain, 'horizontal', h, 'vertical', v, ['z1', 'z2'],
                               z, xunit='mV', yunit='Hz', zunit='A')

    def test_dataset2dplain_shape_measuredata_list_nok2(self):
        v = [0, 1, 2.]
        h = [0., 1, 2, 3, 4]
        measurement = np.arange(len(v) * len(h)).reshape((len(v), len(h)))
        z = [measurement, measurement]
        self.assertRaisesRegex(ValueError, 'The number of measurement names 3 does not match the number of measurements 2',
                               qtt.data.makeDataSet2Dplain, 'horizontal', h, 'vertical', v, ['z1', 'z2', 'z3'],
                               z, xunit='mV', yunit='Hz', zunit='A')

    def test_dataset2dplain_shape_measuredata_ok(self):
        v = [0, 1, 2.]
        h = [0., 1, 2, 3, 4]
        z = np.arange(len(v) * len(h)).reshape((len(v), len(h)))
        data_set = qtt.data.makeDataSet2Dplain('horizontal', h, 'vertical', v, 'z',
                                               z, xunit='mV', yunit='Hz', zunit='A')
        # check attribute
        self.assertTrue(np.array_equal(data_set.z, z))
        # check array
        self.assertTrue(np.array_equal(data_set.arrays['z'], z))

    def test_dataset2dplain_shape_measuredata_nok(self):
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            stream_handler = logging.StreamHandler(sys.stdout)
            logger.addHandler(stream_handler)

            _ = qtt.data.makeDataSet2Dplain('horizontal', [0., 1, 2, 3], 'vertical', [0, 1, 2.], 'z',
                                            np.arange(3 * 5).reshape((3, 5)), xunit='mV', yunit='Hz', zunit='A')

            # Verify warning
            print_string = mock_stdout.getvalue()
            self.assertRegex(print_string, 'Shape of measured data .* does not match setpoint shape .*' )
            logger.removeHandler(stream_handler)

    def test_dataset2dplain_shape_measuredata_second_nok(self):
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            stream_handler = logging.StreamHandler(sys.stdout)
            logger.addHandler(stream_handler)

            _ = qtt.data.makeDataSet2Dplain('horizontal', [0., 1, 2, 3], 'vertical', [0, 1, 2.], ['z1', 'z2'],
                                            [np.arange(3 * 4).reshape((3, 4)), np.arange(3 * 5).reshape((3, 5))],
                                            xunit='mV', yunit='Hz', zunit='A')

            # Verify warning
            print_string = mock_stdout.getvalue()
            self.assertRegex(print_string, 'Shape of measured data .* does not match setpoint shape .*' )
            logger.removeHandler(stream_handler)

    def test_dataset2dplain_no_measuredata(self):
        v = [0, 1, 2.]
        h = [0., 1, 2, 3, 4]
        data_set = qtt.data.makeDataSet2Dplain('horizontal', h, 'vertical', v,
                                               xunit='mV', yunit='Hz', zunit='A')
        # check attribute
        self.assertTrue(data_set.measured.shape == np.ones((len(v), len(h))).shape)
        # check array
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
        self.assertTrue(np.array_equal(data_set.dummy1, x))
        self.assertTrue(data_set.dummy2.shape, data_set.measured.shape)
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

        # check attribute
        self.assertTrue(np.array_equal(data_set.dummy1, x))
        self.assertTrue(data_set.dummy2.shape, data_set.measured.shape)
        self.assertTrue(data_set.measured.shape == np.ones((len(x), len(y))).shape)
        # check array
        self.assertTrue(data_set.arrays['measured'].shape == np.ones((len(x), len(y))).shape)
        # check return names
        self.assertTrue(tuple_names[0][0] == 'dummy1')
        self.assertTrue(tuple_names[0][1] == 'dummy2')
        self.assertTrue(tuple_names[1][0] == 'measured')

    def test_makedataset2d_preset_data_no_list(self):
        p1 = ManualParameter('dummy1')
        p2 = ManualParameter('dummy2')
        x = p1[0:10:1]
        y = p2[0:4:1]
        m1 = ManualParameter('measurement1')
        measure_names = [m1]
        preset_data = np.ones((len(x), len(y)))
        self.assertRaisesRegex(ValueError, 'The number of measurement names 1 does not match the number of measurements 10',
                               qtt.data.makeDataSet2D, x, y, measure_names, preset_data=preset_data, return_names=False)

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

        # check attribute
        self.assertTrue(np.array_equal(data_set.dummy1, x))
        self.assertTrue(np.array_equal(data_set.measurement1, preset_data[0]))
        self.assertTrue(np.array_equal(data_set.measurement2, preset_data[1]))
        # check array
        self.assertTrue(np.array_equal(data_set.arrays['measurement1'], preset_data[0]))
        self.assertTrue(np.array_equal(data_set.arrays['measurement2'], preset_data[1]))

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
        self.assertRaisesRegex(ValueError, 'The number of measurement names 2 does not match the number of measurements 1',
                               qtt.data.makeDataSet2D, x, y, measure_names, preset_data=preset_data, return_names=False)

    def test_makedataset2d_shape_measuredata_list_nok2(self):
        p1 = ManualParameter('dummy1')
        p2 = ManualParameter('dummy2')
        x = p1[0:10:1]
        y = p2[0:4:1]
        measure_names = ['measured1']
        preset_data = [np.ones((len(x), len(y))).shape, np.ones((len(x), len(y))).shape]
        self.assertRaisesRegex(ValueError, 'The number of measurement names 1 does not match the number of measurements 2',
                               qtt.data.makeDataSet2D, x, y, measure_names, preset_data=preset_data, return_names=False)

    def test_makedataset2d_shape_measuredata_nok(self):
        p1 = ManualParameter('dummy1')
        p2 = ManualParameter('dummy2')
        x = p1[0:10:1]
        y = p2[0:4:1]
        measure_names = ['measured1', 'measured2']
        preset_data = [np.ones((len(x) + 1, len(y))), np.ones((len(x), len(y)))]
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            stream_handler = logging.StreamHandler(sys.stdout)
            logger.addHandler(stream_handler)

            _ = qtt.data.makeDataSet2D(x, y, measure_names, preset_data=preset_data, return_names=False)

            # Verify warning
            print_string = mock_stdout.getvalue()
            self.assertRegex(print_string, 'Shape of measured data .* does not match setpoint shape .*' )
            logger.removeHandler(stream_handler)

    def test_makedataset2d_shape_measuredata_second_nok(self):
        p1 = ManualParameter('dummy1')
        p2 = ManualParameter('dummy2')
        x = p1[0:10:1]
        y = p2[0:4:1]
        measure_names = ['measured1', 'measured2']
        preset_data = [np.ones((len(x), len(y))), np.ones((len(x) + 1, len(y)))]
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            stream_handler = logging.StreamHandler(sys.stdout)
            logger.addHandler(stream_handler)

            _ = qtt.data.makeDataSet2D(x, y, measure_names, preset_data=preset_data, return_names=False)

            # Verify warning
            print_string = mock_stdout.getvalue()
            self.assertRegex(print_string, 'Shape of measured data .* does not match setpoint shape .*' )
            logger.removeHandler(stream_handler)
