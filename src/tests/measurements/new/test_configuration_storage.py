import numpy as np
from unittest import mock, TestCase
from unittest.mock import patch, MagicMock

from qtt.measurements.new.configuration_storage import load_configuration, save_configuration
from qilib.utils.serialization import serialize


class TestConfigurationStorage(TestCase):

    def test_load_configuration(self):
        expected_configuration = {'sample_rate': 1000, 'period': 2}
        with patch('builtins.open', mock.mock_open(read_data=serialize(expected_configuration))):
            actual_configuration = load_configuration('/dev/null')
        self.assertEqual(expected_configuration, actual_configuration)

    """
    def test_save_configuration(self):
        adapter = MagicMock()
        adapter.return_value.read.reset_mock = {'sample_rate': 1000, 'period': 2}
        with patch('builtins.open', mock.mock_open()):
            save_configuration('/dev/null', adapter)
    """
