import unittest
from unittest.mock import patch

from qilib.utils import PythonJsonStructure
from qtt.instrument_drivers.adapters import TimeStampInstrumentAdapter


class TestTimeStampInstrumentAdapter(unittest.TestCase):
    def test_read(self):
        with patch('time.time', return_value=42):
            adapter = TimeStampInstrumentAdapter('some_address')
            config = adapter.read()
        self.assertEqual(42, config['timestamp']['value'])
        self.assertEqual(0, config['timestamp_offset']['value'])
        adapter.close_instrument()

    def test_apply(self):
        adapter = TimeStampInstrumentAdapter('other_address')
        config = PythonJsonStructure(timestamp_offset={'value': 42})
        adapter.apply(config)
        read_config = adapter.read()
        self.assertEqual(42, read_config['timestamp_offset']['value'])
        adapter.close_instrument()
