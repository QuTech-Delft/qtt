import base64
import json
from json import JSONDecoder, JSONEncoder

import numpy as np
import qcodes

import qtt.data


class JsonSerializeKey:
    """The custum value types for the JSON serializer."""
    OBJECT = '__object__'
    CONTENT = '__content__'
    DATA_TYPE = '__data_type__'


class QttJsonDecoder(JSONDecoder):

    def __init__(self, *args, **kwargs):
        """ JSON decoder that handles numpy arrays and tuples."""

        super().__init__(object_hook=QttJsonDecoder.__object_hook, *args, **kwargs)

    @staticmethod
    def __decode_bytes(item):
        return base64.b64decode(item[JsonSerializeKey.CONTENT].encode('ascii'))

    @staticmethod
    def __decode_qcodes_instrument(item):
        return item

    @staticmethod
    def __decode_tuple(item):
        return tuple(item[JsonSerializeKey.CONTENT])

    @staticmethod
    def __decode_numpy_number(item):
        obj = item[JsonSerializeKey.CONTENT]
        return np.frombuffer(base64.b64decode(obj['__npnumber__']), dtype=np.dtype(obj['dtype']))[0]

    @staticmethod
    def __decode_numpy_array(item):
        obj = item[JsonSerializeKey.CONTENT]
        array = np.frombuffer(base64.b64decode(obj['__ndarray__']), dtype=np.dtype(obj['dtype'])).reshape(obj['shape'])
        # make the array writable
        array = np.array(array)
        return array

    @staticmethod
    def __decode_qcodes_dataset(item):
        obj = item[JsonSerializeKey.CONTENT]
        return qtt.data.dictionary_to_dataset(obj['__dataset_dictionary__'])

    @staticmethod
    def __object_hook(obj):
        decoders = {
            bytes.__name__: QttJsonDecoder.__decode_bytes,
            tuple.__name__: QttJsonDecoder.__decode_tuple,
            np.array.__name__: QttJsonDecoder.__decode_numpy_array,
            '__npnumber__': QttJsonDecoder.__decode_numpy_number,
            '__qcodes_instrument__': QttJsonDecoder.__decode_qcodes_instrument,
            '__qcodes.DataSet__': QttJsonDecoder.__decode_qcodes_dataset,
        }
        if JsonSerializeKey.CONTENT in obj:
            decoder_function = decoders.get(obj[JsonSerializeKey.OBJECT])
            return decoder_function(obj)
        return obj


class QttJsonEncoder(JSONEncoder):
    """ JSON encoder that handles numpy arrays and tuples """

    @staticmethod
    def __encode_bytes(item):
        return {
            JsonSerializeKey.OBJECT: bytes.__name__,
            JsonSerializeKey.CONTENT: base64.b64encode(item).decode('ascii')
        }

    @staticmethod
    def __encode_qcodes_instrument(item):
        return {
            JsonSerializeKey.OBJECT: '__qcodes_instrument__',
            JsonSerializeKey.CONTENT: {'name': item.name, 'qcodes_instrument': str(item)}
        }

    @staticmethod
    def __encode_tuple(item):
        return {
            JsonSerializeKey.OBJECT: tuple.__name__,
            JsonSerializeKey.CONTENT: [QttJsonEncoder.__encoder(value) for value in item]
        }

    @staticmethod
    def __encode_list(item):
        return [QttJsonEncoder.__encoder(value) for value in item]

    @staticmethod
    def __encode_dict(item):
        return {
            key: QttJsonEncoder.__encoder(value) for key, value in item.items()
        }

    @staticmethod
    def __encode_numpy_number(item):
        return {
            JsonSerializeKey.OBJECT: '__npnumber__',
            JsonSerializeKey.CONTENT: {
                '__npnumber__': base64.b64encode(item.tobytes()).decode('ascii'),
                'dtype': item.dtype.str,
            }
        }

    @staticmethod
    def __encode_numpy_array(item):
        return {
            JsonSerializeKey.OBJECT: np.array.__name__,
            JsonSerializeKey.CONTENT: {
                '__ndarray__': base64.b64encode(item.tobytes()).decode('ascii'),
                'dtype': item.dtype.str,
                'shape': item.shape,
            }
        }

    @staticmethod
    def __encode_qcodes_dataset(item):
        return {
            JsonSerializeKey.OBJECT: '__qcodes.DataSet__',
            JsonSerializeKey.CONTENT: {
                '__dataset_dictionary__': QttJsonEncoder.__encoder(qtt.data.dataset_to_dictionary(item)),
            }
        }

    @staticmethod
    def __encoder(item):
        encoders = {
            bytes: QttJsonEncoder.__encode_bytes,
            tuple: QttJsonEncoder.__encode_tuple,
            list: QttJsonEncoder.__encode_list,
            dict: QttJsonEncoder.__encode_dict,
            np.ndarray: QttJsonEncoder.__encode_numpy_array,
            np.int32: QttJsonEncoder.__encode_numpy_number,
            np.int64: QttJsonEncoder.__encode_numpy_number,
            np.float32: QttJsonEncoder.__encode_numpy_number,
            np.float64: QttJsonEncoder.__encode_numpy_number,
            qcodes.Instrument: QttJsonEncoder.__encode_qcodes_instrument,
            qcodes.DataSet: QttJsonEncoder.__encode_qcodes_dataset,
        }
        encoder_function = encoders.get(type(item), None)
        return encoder_function(item) if encoder_function else item

    def encode(self, o):
        return super().encode(QttJsonEncoder.__encoder(o))


def encode_json(data: object) -> str:
    """ Encode Python object to JSON

    Args:
        data: data to be encoded
    Returns
        String with formatted JSON

    """
    return json.dumps(data, cls=QttJsonEncoder, indent=2)


def decode_json(json_string: str) -> object:
    """ Decode Python object to JSON

    Args:
        json_string: data to be decoded
    Returns
        Python object

    """
    return json.loads(json_string, cls=QttJsonDecoder)


def save_json(data: object, filename: str):
    """ Write a Python object to a JSON file

    Args:
        data (object): object to be serialized
        filename (str): filename to write data to
    """
    with open(filename, 'wt') as fid:
        fid.write(encode_json(data))


def load_json(filename: str) -> object:
    """ Write a Python object from a JSON file

    Args:
        filename (str): filename to write data to
    Returns:
        object: object loaded from JSON file
    """
    with open(filename, 'rt') as fid:
        data = fid.read()
    return decode_json(data)
