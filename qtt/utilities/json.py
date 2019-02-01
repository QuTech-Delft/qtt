import numpy as np
import base64
import json

from json import JSONDecoder, JSONEncoder

class JsonSerializeKey:
    """The custum value types for the JSON serializer."""
    OBJECT = '__object__'
    CONTENT = '__content__'
    DATA_TYPE = '__data_type__'
    
class CustomJsonDecoder(JSONDecoder):

    def __init__(self, *args, **kwargs):
        """ JSON decoder that handles numpy arrays and tuples """
        
        super().__init__(object_hook=CustomJsonDecoder.__object_hook, *args, **kwargs)

    @staticmethod
    def __decode_bytes(item):
        return base64.b64decode(item[JsonSerializeKey.CONTENT].encode('ascii'))

    @staticmethod
    def __decode_tuple(item):
        return tuple(item[JsonSerializeKey.CONTENT])


    @staticmethod
    def __decode_numpy_number(item):
        obj = item[JsonSerializeKey.CONTENT]       
        return np.frombuffer(
                base64.b64decode(obj['__npnumber__']),
                dtype=np.dtype(obj['dtype'])
            )[0]

    @staticmethod
    def __decode_numpy_array(item):
        obj = item[JsonSerializeKey.CONTENT]
        return np.frombuffer(
                base64.b64decode(obj['__ndarray__']),
                dtype=np.dtype(obj['dtype'])
            ).reshape(obj['shape'])
        
    @staticmethod
    def __object_hook(obj):
        decoders = {
            bytes.__name__: CustomJsonDecoder.__decode_bytes,
            tuple.__name__: CustomJsonDecoder.__decode_tuple,
            np.array.__name__: CustomJsonDecoder.__decode_numpy_array,
            '__npnumber__': CustomJsonDecoder.__decode_numpy_number,
        }
        if JsonSerializeKey.CONTENT in obj:
            decoder_function = decoders.get(obj[JsonSerializeKey.OBJECT])
            return decoder_function(obj)
        return obj
    



class CustomJsonEncoder(JSONEncoder):
    """ JSON encoder that handles numpy arrays and tuples """

    @staticmethod
    def __encode_bytes(item):
        return {
            JsonSerializeKey.OBJECT: bytes.__name__,
            JsonSerializeKey.CONTENT: base64.b64encode(item).decode('ascii')
        }

    @staticmethod
    def __encode_tuple(item):
        return {
            JsonSerializeKey.OBJECT: tuple.__name__,
            JsonSerializeKey.CONTENT: [CustomJsonEncoder.__encoder(value) for value in item]
        }

    @staticmethod
    def __encode_list(item):
        return [CustomJsonEncoder.__encoder(value) for value in item] 

    @staticmethod
    def __encode_dict(item):
        return {
            key: CustomJsonEncoder.__encoder(value) for key, value in item.items()
        }
           
    @staticmethod
    def __encode_numpy_number(item):
        return {
            JsonSerializeKey.OBJECT: '__npnumber__',
            JsonSerializeKey.CONTENT:  {
                '__npnumber__': base64.b64encode(item.tobytes()).decode('ascii'),
                'dtype': item.dtype.str,
            }
        }

    @staticmethod
    def __encode_numpy_array(item):
        return {
            JsonSerializeKey.OBJECT: np.array.__name__,
            JsonSerializeKey.CONTENT:  {
                '__ndarray__': base64.b64encode(item.tobytes()).decode('ascii'),
                'dtype': item.dtype.str,
                'shape': item.shape,
            }
        }

    @staticmethod
    def __encoder(item):
        encoders = {
            bytes: CustomJsonEncoder.__encode_bytes,
            tuple: CustomJsonEncoder.__encode_tuple,
            list: CustomJsonEncoder.__encode_list,
            dict: CustomJsonEncoder.__encode_dict,
            np.ndarray: CustomJsonEncoder.__encode_numpy_array,
            np.int32: CustomJsonEncoder.__encode_numpy_number,
            np.int64: CustomJsonEncoder.__encode_numpy_number,
            np.float32: CustomJsonEncoder.__encode_numpy_number,
            np.float64: CustomJsonEncoder.__encode_numpy_number,
        }
        encoder_function = encoders.get(type(item), None)
        return encoder_function(item) if encoder_function else item

    def encode(self, o):
        return super().encode(CustomJsonEncoder.__encoder(o))    

def save_json(data, filename):
    with open(filename, 'wt') as fid:
        fid.write(json.dumps(data, cls=CustomJsonEncoder, indent=2))
        
def load_json(filename):
    with open(filename, 'rt') as fid:
        data = fid.read()
    return json.loads(data, cls=CustomJsonDecoder)
    
import unittest

class TestJSON(unittest.TestCase):
    def test_custom_encoders(self):
        data = {'float': 1.0, 'str': 'hello', 'tuple': (1,2,3)}
        serialized_data = json.dumps(data, cls=CustomJsonEncoder)
        
        loaded_data = json.loads(serialized_data, cls=CustomJsonDecoder)
    
        self.assertDictEqual(data, loaded_data)

    def test_numpy_encoders(self):
        data = {'array': np.array([1.,2,3]), 'intarray': np.array([1,2])}
        serialized_data = json.dumps(data, cls=CustomJsonEncoder)
        loaded_data = json.loads(serialized_data, cls=CustomJsonDecoder)
    
        for key in data.keys():
            np.testing.assert_array_equal(data[key], loaded_data[key])
    
        data = {'int32': np.int32(2.), 'float': np.float(3.), 'float64': np.float64(-1)}
        serialized_data = json.dumps(data, cls=CustomJsonEncoder)
        loaded_data = json.loads(serialized_data, cls=CustomJsonDecoder)
    
if __name__=='__main__':
    unittest.main()    