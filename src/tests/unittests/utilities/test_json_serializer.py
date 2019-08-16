import unittest
import json
import numpy as np

import qcodes
import qcodes.tests.data_mocks

from qtt.utilities.json_serializer import encode_json, decode_json


class TestJSONSerializer(unittest.TestCase):

    def test_custom_encoders(self):
        data = {'float': 1.0, 'str': 'hello', 'tuple': (1, 2, 3)}
        serialized_data = encode_json(data)
        loaded_data = decode_json(serialized_data)
        self.assertDictEqual(data, loaded_data)

    def test_qcodes_dataset_encoding(self):
        dataset = qcodes.tests.data_mocks.DataSet2D()

        json_data = encode_json(dataset)
        self.assertIsInstance(json_data, str)
        dataset2 = decode_json(json_data)
        self.assertIsInstance(dataset2, qcodes.DataSet)

    def test_numpy_encoders(self):
        data = {'array': np.array([1., 2, 3]), 'intarray': np.array([1, 2])}
        serialized_data = encode_json(data)
        loaded_data = decode_json(serialized_data)
        self.assertIn('__ndarray__', serialized_data)
        self.assertIn(r'"__data_type__": "<f8"', serialized_data)
        np.testing.assert_array_equal(loaded_data['array'], data['array'])

        for key, value in data.items():
            np.testing.assert_array_equal(value, loaded_data[key])

        data = {'int32': np.int32(2.), 'float': np.float(3.), 'float64': np.float64(-1)}
        serialized_data = encode_json(data)
        loaded_data = decode_json(serialized_data)

    def test_numpy_array_writable(self):
        data = {'array': np.array([1., 2, 3]), 'intarray': np.array([1, 2])}
        serialized_data = encode_json(data)
        loaded_data = decode_json(serialized_data)
        self.assertTrue(loaded_data['array'].flags.writeable)
        self.assertTrue(loaded_data['intarray'].flags.writeable)

    def test_numpy_bool_type(self):
        data = {'true': np.bool_(True), 'false': np.bool_(False)}
        serialized_data = encode_json(data)
        loaded_data = decode_json(serialized_data)
        self.assertIn('{"true": {"__object__": "__npnumber__", "__content__"', serialized_data)
        self.assertEqual(loaded_data['true'], np.bool_(True))

    def test_instrument_encoders(self):
        instrument = qcodes.Instrument('test_instrument_y')
        data = {'instrument': instrument}
        serialized_data = encode_json(data)
        loaded_data = decode_json(serialized_data)
        self.assertEqual(loaded_data['instrument']['__object__'], '__qcodes_instrument__')
        self.assertEqual(loaded_data['instrument']['__content__']['name'], instrument.name)
        instrument.close()
