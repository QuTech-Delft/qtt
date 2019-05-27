import os
import unittest
import unittest.mock
import io

import qcodes
import qcodes.tests.data_mocks

import qtt.data
import numpy as np
import matplotlib.pyplot as plt


class TestDataSet(unittest.TestCase):
    """ Integration tests for DataSet used by qtt.

    DataSet requirements:

        - Contains arrays with metadata such as units, label
        - Live plotting of data
        - Information about dependent and independent variables
        - Storage to disk working with a location formatter
        - Load legacy DataSet forms and convert to current DataSet
        - Incremental adding of data to the arrays
        - Serialization
        - Utility functions:
            * default_parameter_array()
            * qcodes.MatPlot, qcodes.QtPlot
            * Easy conversion to numpy: array = np.array(dataset.voltage). Even better: the DataArray's should
                support the __array_interface__ attribute.

    """

    def setUp(self):
        dataset_class = qcodes.DataSet
        self.dataset_class = dataset_class
        self.dataset1d = qcodes.tests.data_mocks.DataSet1D()
        self.dataset1d.metadata['hello'] = 'world'
        self.dataset2d = qtt.data.makeDataSet2Dplain('x', [0, 1, 2, 3.], 'y', [3, 4, 5, 6.], xunit='mV', yunit='a.u.',
                                                     zunit='a.u.')

    def test_dataset_has_default_parameter_array(self):
        """ Each DataSet has a method to provide the "main" array."""
        data_array = self.dataset2d.default_parameter_array()
        self.assertEqual(data_array.name, 'measured')
        self.assertEqual(data_array.set_arrays, (self.dataset2d.y, self.dataset2d.x,))

    def test_dataarray_has_metadata(self):
        """ Each DataSet has support to store metadata."""
        data_array = self.dataset2d.default_parameter_array()

        unit = data_array.unit
        self.assertEqual(unit, 'a.u.')

    def test_load_legacy_dataset(self):
        """ We need to convert old datasets to the current dataset structure."""
        exampledatadir = os.path.join(qtt.__path__[0], 'exampledata')
        qcodes.DataSet.default_io = qcodes.DiskIO(exampledatadir)
        old_dataset = qtt.data.load_dataset(os.path.join('2017-09-04', '11-05-17_qtt_scan2Dfastvec'))

        def convert_legacy(old_dataset):
            """ Dummy converter """
            return old_dataset

        new_dataset = convert_legacy(old_dataset)
        self.assertIsInstance(new_dataset, qcodes.DataSet)

    def test_metadata(self):
        """ Test a dataset has metadata that can be get and set."""
        self.dataset1d.metadata.keys()

        expected = {'a': 1, 'b': np.array([1, 1])}
        self.dataset1d.metadata['new_metadata'] = expected

        self.assertDictEqual(self.dataset1d.metadata.get('new_metadata'), expected)
        with self.assertRaises(KeyError):
            _ = self.dataset1d.metadata['non_existing_metadata']

    def test_location_provider(self):
        """ The DataSet generated locations (or tags) for storage automatically. The format is user configurable."""
        location = self.dataset2d.location_provider(qcodes.DataSet.default_io)
        self.assertIsInstance(location, str)

    def test_serialization(self):
        """ A DataSet needs to be serialized to a simple Python dictionary."""
        dataset_dictionary = qtt.data.dataset_to_dictionary(self.dataset2d)
        self.assertIsInstance(dataset_dictionary, dict)
        dataset2 = qtt.data.dictionary_to_dataset(dataset_dictionary)
        self.assertEqual(list(dataset2.arrays), ['measured', 'x', 'y'])

    def test_dataset_storage(self):
        """ A DataSet (including metadata) can be stored to and loaded from disk.

        We need a binary format for efficiency.
        Incremental storage is a nice-to-have (and currently implemented).

        """
        self.dataset2d.write()

        loaded_dataset = qtt.data.load_dataset(self.dataset2d.location)
        self.assertEqual(sorted(list(loaded_dataset.arrays.keys())), sorted(list(self.dataset2d.arrays.keys())))

    def test_has_representation(self):
        """ Each dataset needs to have a presentation that quickly shows the content of the DataSet."""

        with unittest.mock.patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            print(self.dataset1d)
        print_string = mock_stdout.getvalue()
        lines = print_string.split('\n')
        self.assertEqual(lines[0], 'DataSet:')
        self.assertEqual(lines[3], '   Measured | y          | y            | (5,)')

    def test_plot_matplotlib(self):
        """ We need to plot simple 1D and 2D datasets with proper units and labels."""
        plt.close(100)
        xarray = self.dataset1d.x_set
        qcodes.MatPlot(self.dataset1d.default_parameter_array(), num=100)
        plt.figure(100)
        ax = plt.gca()
        self.assertEqual(ax.xaxis.label.get_text(), xarray.label + ' (' + str(xarray.unit) + ')')
