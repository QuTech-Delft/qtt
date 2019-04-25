""" Functionality to store and load instrument configurations."""

from qilib.utils import PythonJsonStructure, serialization


def load_configuration(file_path: str) -> PythonJsonStructure:
    """ Loads the instrument configuration from disk storage.

    Args:
        file_path: The store file location on disk.

    Returns:
        The loaded configuration from disk.
    """
    with open(file_path, 'rb') as file_pointer:
        serialized_configuration = file_pointer.readlines()
    unserialized_configuration = dict(serialization.unserialize(serialized_configuration[0]))
    return PythonJsonStructure(unserialized_configuration)


def save_configuration(file_path: str, configuration: PythonJsonStructure) -> None:
    """ Saves the instrument configuration to disk storage.

    Args:
        file_path: The store file location on disk.
        configuration: The instrument configuration that needs to be stored to disk.
    """
    with open(file_path, 'wb') as file_pointer:
        serialized_configuration = serialization.serialize(configuration)
        file_pointer.write(serialized_configuration)
