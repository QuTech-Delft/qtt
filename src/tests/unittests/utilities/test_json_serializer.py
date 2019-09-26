import unittest

import numpy as np
import qcodes
import qcodes.tests.data_mocks

from qtt.utilities.json_serializer import encode_json, decode_json, qtt_serializer, JsonSerializeKey, encode_json_dataclass, decode_json_dataclass


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

    def test_float_nan_inf(self):
        data = [np.NaN, np.Inf, 1.]
        json_data = encode_json(data)
        self.assertIn('NaN', json_data)
        loaded_data = decode_json(json_data)
        self.assertTrue(np.isnan(loaded_data[0]))
        self.assertTrue(np.isinf(loaded_data[1]))
        self.assertEqual(loaded_data[2], 1.)

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

    def test_old_format_numpy_array(self):
        data = np.array([1, 2, 3, 4, 5])
        encoded_old = {'__object__': 'array',
                       '__content__': {'__ndarray__': 'AQAAAAIAAAADAAAABAAAAAUAAAA=', 'shape': [5], 'dtype': '<i4'}}

        self.assertIsNone(np.testing.assert_array_equal(data, qtt_serializer.decode_data(encoded_old)))

    def test_old_format_numpy_number(self):
        data = np.float32(13.37)
        encoded_old = {'__object__': '__npnumber__', '__content__': {'__npnumber__': 'hetVQQ==', 'dtype': '<f4'}}

        self.assertIsNone(np.testing.assert_almost_equal(data, qtt_serializer.decode_data(encoded_old), 1))

    def test_encode_json_dataclass(self):
        tag='123'
        mock_object = unittest.mock.MagicMock()
        mock_object.to_dict.return_value = 'mock_return_value'

        d=encode_json_dataclass(tag, mock_object)
        self.assertEqual(d[JsonSerializeKey.OBJECT], tag)
        self.assertEqual(d[JsonSerializeKey.CONTENT], 'mock_return_value')

    def test_decode_json_dataclass(self):
        mock_class = unittest.mock.MagicMock()
        mock_class.from_dict.return_value = 'decoded'
        mock_object = unittest.mock.MagicMock()

        d=decode_json_dataclass(mock_class, mock_object)
        mock_class.from_dict.assert_called_once_with(mock_object[JsonSerializeKey.CONTENT])
        self.assertEqual(d, 'decoded')
