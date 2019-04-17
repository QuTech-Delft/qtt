from unittest import TestCase
from unittest.mock import patch, MagicMock, mock_open, call

from qtt.measurements.acquisition.configuration_storage import load_configuration, save_configuration
from qilib.utils.serialization import serialize


class TestConfigurationStorage(TestCase):

    def test_load_configuration(self):
        expected_configuration = {'sample_rate': 1000, 'period': 2}
        with patch('builtins.open', mock_open(read_data=serialize(expected_configuration))):
            actual_configuration = load_configuration('/dev/null')
        self.assertEqual(expected_configuration, actual_configuration)

    def test_save_configuration(self):
        file_path = '/dev/null'
        configuration = {'sample_rate': 1000, 'period': 2}

        with patch('builtins.open', new_callable=mock_open) as file_mock:
            save_configuration(file_path, configuration)

        file_mock.return_value.write.assert_called_once_with(serialize(configuration))
