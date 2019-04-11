import unittest
from unittest import TestCase

import qtt.measurements.scans

# %%


class TestTimeStampInstrument(TestCase):

   def test_TimeStampInstrument(self):
    timestamp_instrument = qtt.instrument_drivers.TimeStamp.TimeStampInstrument(qtt.measurements.scans.instrumentName('timestamp'))
    assert(isinstance(timestamp_instrument.timestamp(), float))



if __name__ == '__main__':
    unittest.main()
