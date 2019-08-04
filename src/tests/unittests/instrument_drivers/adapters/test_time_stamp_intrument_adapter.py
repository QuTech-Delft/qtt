import unittest
from qtt.instrument_drivers.adapters import TimeStampInstrumentAdapter


class TestTimeStampInstrumentAdapter(unittest.TestCase):
    def test_read_returns_empty_pjs(self):
        adapter = TimeStampInstrumentAdapter('some_address')
        config = adapter.read()
        self.assertDictEqual({}, config)
        adapter.close_instrument()
