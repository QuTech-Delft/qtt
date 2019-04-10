from qilib.configuration_helper import InstrumentAdapter
from qilib.utils import PythonJsonStructure, serialization


def load_configuration(file_path: str) -> PythonJsonStructure:
    """ Loads the instrument configuration from storage.

    Args:
        file_path: The store file location on disk.

    Returns:
        The loaded configuration from disk.
    """
    with open(file_path, 'rb') as file_pointer:
        string_data = file_pointer.readlines()
    serialized_data = dict(serialization.unserialize(string_data[0]))
    return PythonJsonStructure(serialized_data)


def save_configuration(file_path: str, adapter: InstrumentAdapter):
    """ Saves the instrument configuration to disk given the instrument adapter.

    Args:
        file_path: The store file location on disk.
        adapter: The instrument adapter instance which configuration needs to be stored.
    """
    configuration = adapter.read()
    with open(file_path, 'wb+') as file_pointer:
        serialized_data = serialization.serialize(configuration)
        file_pointer.write(serialized_data)
