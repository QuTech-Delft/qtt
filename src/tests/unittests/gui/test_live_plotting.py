import unittest
import numpy as np
import pyqtgraph
import unittest.mock as mock
import io

import qtt
from qtt.measurements.scans import instrumentName
from qtt.gui.live_plotting import MockCallback_2d, livePlot, MeasurementControl


class TestLivePlotting(unittest.TestCase):

    def test_mock2d(self):
        np.random.seed(2019)
        mock_callback = MockCallback_2d(instrumentName('dummy2d'))
        data_reshaped = mock_callback()
        mock_callback.close()

        self.assertEqual(data_reshaped.size, 6400)
        self.assertAlmostEqual(data_reshaped.min(), 2.996900483446681e-05, 6)
        self.assertAlmostEqual(data_reshaped.max(), 4.122574793363507, 6)
        self.assertAlmostEqual(data_reshaped.sum(), 3440.344282085355, 3)
