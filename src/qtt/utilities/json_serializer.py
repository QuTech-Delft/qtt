import base64
import numpy as np
import qcodes
from qilib.data_set.mongo_data_set_io import NumpyKeys

import qtt.data
from typing import Any
from qilib.utils.serialization import Serializer, JsonSerializeKey, serializer


class QttSerializer(Serializer):
    def __init__(self):
        super().__init__()

        self.register(qcodes.Instrument, encode_qcodes_instrument, '__qcodes_instrument__',
                      decode_qcodes_instrument)
        for numpy_integer_type in [np.int32, np.int64, np.float32, np.float64, np.bool_]:
            self.register(numpy_integer_type, encode_numpy_number, '__npnumber__', decode_numpy_number)
        self.register(qcodes.DataSet, encode_qcodes_dataset, '__qcodes_dataset__', decode_qcodes_dataset)
        self.register(np.ndarray, encode_numpy_array, np.array.__name__, decode_numpy_array)


def encode_qcodes_instrument(item):
    return {
        JsonSerializeKey.OBJECT: '__qcodes_instrument__',
        JsonSerializeKey.CONTENT: {'name': item.name, 'qcodes_instrument': str(item)}
    }


def decode_qcodes_instrument(item):
    return item


def encode_numpy_number(item):
    return {
        JsonSerializeKey.OBJECT: '__npnumber__',
        JsonSerializeKey.CONTENT: {
            '__npnumber__': base64.b64encode(item.tobytes()).decode('ascii'),
            'dtype': item.dtype.str,
        }
    }


def decode_numpy_number(item):
    obj = item[JsonSerializeKey.CONTENT]
    return np.frombuffer(base64.b64decode(obj['__npnumber__']), dtype=np.dtype(obj['dtype']))[0]


def encode_qcodes_dataset(item):
    dataset_dictionary = qtt_serializer.encode_data(qtt.data.dataset_to_dictionary(item))
    return {
        JsonSerializeKey.OBJECT: '__qcodes_dataset__',
        JsonSerializeKey.CONTENT: {
            '__dataset_dictionary__': dataset_dictionary,
        }
    }


def decode_qcodes_dataset(item):
    obj = item[JsonSerializeKey.CONTENT]
    return qtt.data.dictionary_to_dataset(obj['__dataset_dictionary__'])


def encode_numpy_array(item):
    return serializer.encode_data(item)


def decode_numpy_array(item):
    if 'dtype' in item[JsonSerializeKey.CONTENT]:
        item[JsonSerializeKey.CONTENT][NumpyKeys.DATA_TYPE] = item[JsonSerializeKey.CONTENT].pop('dtype')
    if 'shape' in item[JsonSerializeKey.CONTENT]:
        item[JsonSerializeKey.CONTENT][NumpyKeys.SHAPE] = item[JsonSerializeKey.CONTENT].pop('shape')

    return serializer.decode_data(item)


def encode_json(data: object) -> str:
    """ Encode Python object to JSON

    Args:
        data: data to be encoded
    Returns
        String with formatted JSON

    """
    return qtt_serializer.serialize(data)


def decode_json(json_string: str) -> Any:
    """ Decode Python object to JSON

    Args:
        json_string: data to be decoded
    Returns
        Python object

    """
    return qtt_serializer.unserialize(json_string)


def save_json(data: Any, filename: str):
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


qtt_serializer = QttSerializer()
