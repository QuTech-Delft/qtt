import unittest
from unittest import TestCase
from unittest.mock import patch

import qtt.measurements.scans
import qtt.instrument_drivers.TimeStamp

# %%


class TestTimeStampInstrument(TestCase):

    def setUp(self):
        self.timestamp_instrument = qtt.instrument_drivers.TimeStamp.TimeStampInstrument(
            qtt.measurements.scans.instrumentName('timestamp'))

    def tearDown(self):
        self.timestamp_instrument.close()

    @patch('time.time', return_value=0.124)
    def test_TimeStampInstrument(self, time_method):
        timestamp = self.timestamp_instrument.timestamp()
        assert(isinstance(timestamp, float))
        self.assertEqual(timestamp, 0.124)
        time_method.assert_called_with()

    def test_TimeStampInstrument_offset_initial_value(self):
        timestamp_instrument = self.timestamp_instrument
        offset = timestamp_instrument.timestamp_offset()
        self.assertEqual(offset, 0)

    def test_TimeStampInstrument_offset(self):
        timestamp = self.timestamp_instrument.timestamp()
        self.timestamp_instrument.timestamp_offset(timestamp)
        timestamp = self.timestamp_instrument.timestamp()
        self.assertTrue(timestamp >= 0)
        self.assertTrue(timestamp <= 1)


if __name__ == '__main__':
    unittest.main()
