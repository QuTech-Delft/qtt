import base64
from typing import Any

import numpy as np
import qcodes
from qilib.utils.serialization import serialize, unserialize, register_encoder, register_decoder, transform_data, \
    JsonSerializeKey

import qtt.data


def encode_tuple(item):
    return {
        JsonSerializeKey.OBJECT: tuple.__name__,
        JsonSerializeKey.CONTENT: [transform_data(value) for value in item]
    }


def decode_tuple(item):
    return tuple(item[JsonSerializeKey.CONTENT])


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
    dataset_dictionary = transform_data(qtt.data.dataset_to_dictionary(item))
    return {
        JsonSerializeKey.OBJECT: '__qcodes_dataset__',
        JsonSerializeKey.CONTENT: {
            '__dataset_dictionary__': dataset_dictionary,
        }
    }


def decode_qcodes_dataset(item):
    obj = item[JsonSerializeKey.CONTENT]
    return qtt.data.dictionary_to_dataset(obj['__dataset_dictionary__'])


def encode_json(data: object) -> str:
    """ Encode Python object to JSON

    Args:
        data: data to be encoded
    Returns
        String with formatted JSON

    """
    return serialize(data)


def decode_json(json_string: str) -> Any:
    """ Decode Python object to JSON

    Args:
        json_string: data to be decoded
    Returns
        Python object

    """
    return unserialize(json_string)


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


register_encoder(tuple, encode_tuple)
register_decoder(tuple.__name__, decode_tuple)
register_encoder(qcodes.Instrument, encode_qcodes_instrument)
register_decoder('__qcodes_instrument__', decode_qcodes_instrument)
for t in [np.int32, np.int64, np.float32, np.float64, np.bool_]:
    register_encoder(t, encode_numpy_number)
register_decoder('__npnumber__', decode_numpy_number)
register_encoder(qcodes.DataSet, encode_qcodes_dataset)
register_decoder('__qcodes_dataset__', decode_qcodes_dataset)
