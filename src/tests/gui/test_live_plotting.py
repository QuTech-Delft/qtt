
import unittest
from qtt.measurements.scans import instrumentName
from qtt.gui.live_plotting import MockCallback_2d


class TestLivePlotting(unittest.TestCase):

    def test_mock2d(self):
        mock_callback = MockCallback_2d(instrumentName('dummy2d'))
        mock_callback()

        mock_callback.close()
