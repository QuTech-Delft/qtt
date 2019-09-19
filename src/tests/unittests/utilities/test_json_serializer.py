import unittest
import json
import numpy as np

import qcodes
import qcodes.tests.data_mocks
from qtt.utilities.json_serializer import QttJsonEncoder, QttJsonDecoder, encode_json, decode_json


class TestJSONSerializer(unittest.TestCase):

    def test_custom_encoders(self):
        data = {'float': 1.0, 'str': 'hello', 'tuple': (1, 2, 3)}
        serialized_data = json.dumps(data, cls=QttJsonEncoder)

        loaded_data = json.loads(serialized_data, cls=QttJsonDecoder)
        self.assertDictEqual(data, loaded_data)

    def test_qcodes_dataset_encoding(self):
        dataset = qcodes.tests.data_mocks.DataSet2D()

        json_data = encode_json(dataset)
        self.assertIsInstance(json_data, str)
        dataset2 = decode_json(json_data)
        self.assertIsInstance(dataset2, qcodes.DataSet)

    def test_float_nan_inf(self):
        data = [np.NaN, np.Inf, 1.]
        json_data = encode_json(data)
        self.assertIn('NaN', json_data)
        loaded_data = decode_json(json_data)
        self.assertTrue(np.isnan(loaded_data[0]))
        self.assertTrue(np.isinf(loaded_data[1]))
        self.assertEqual(loaded_data[2], 1.)

    def test_numpy_bool_type(self):
        data = {'true': np.bool_(True), 'false': np.bool_(False)}
        serialized_data = json.dumps(data, cls=QttJsonEncoder)
        loaded_data = json.loads(serialized_data, cls=QttJsonDecoder)
        self.assertIn('{"true": {"__object__": "__npnumber__", "__content__"', serialized_data)
        self.assertEqual(loaded_data['true'], np.bool_(True))

    def test_numpy_encoders(self):
        data = {'array': np.array([1., 2, 3]), 'intarray': np.array([1, 2])}
        serialized_data = json.dumps(data, cls=QttJsonEncoder)
        loaded_data = json.loads(serialized_data, cls=QttJsonDecoder)
        self.assertIn('__ndarray__', serialized_data)
        self.assertIn(r'"dtype": "<f8"', serialized_data)
        np.testing.assert_array_equal(loaded_data['array'], data['array'])

        for key, value in data.items():
            np.testing.assert_array_equal(value, loaded_data[key])

        data = {'int32': np.int32(2.), 'float': np.float(3.), 'float64': np.float64(-1)}
        serialized_data = json.dumps(data, cls=QttJsonEncoder)
        loaded_data = json.loads(serialized_data, cls=QttJsonDecoder)

    def test_numpy_array_writable(self):
        data = {'array': np.array([1., 2, 3]), 'intarray': np.array([1, 2])}
        serialized_data = json.dumps(data, cls=QttJsonEncoder)
        loaded_data = json.loads(serialized_data, cls=QttJsonDecoder)
        self.assertTrue(loaded_data['array'].flags['WRITEABLE'])
        self.assertTrue(loaded_data['intarray'].flags['WRITEABLE'])

    def test_instrument_encoders(self):
        instrument = qcodes.Instrument('test_instrument_y')
        data = {'instrument': instrument}
        serialized_data = json.dumps(data, cls=QttJsonEncoder)
        loaded_data = json.loads(serialized_data, cls=QttJsonDecoder)
        self.assertEqual(loaded_data['instrument']['__object__'], '__qcodes_instrument__')
        self.assertEqual(loaded_data['instrument']['__content__']['name'], instrument.name)
        instrument.close()

