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


    def test_livePlot(self):
        _ = pyqtgraph.mkQApp()

        mock_callback = MockCallback_2d(qtt.measurements.scans.instrumentName('mock'))
        lp = livePlot(datafunction=mock_callback, sweepInstrument=None,
                      sweepparams=['L', 'R'], sweepranges=[50, 50], show_controls=True)
        lp.win.setGeometry(1500, 10, 400, 400)
        lp.startreadout()
        lp.crosshair(True)
        lp.stopreadout()
        lp.updatebg()
        lp.close()
        self.assertIsInstance(lp.datafunction_result, np.ndarray)

        mock_callback.close()


class TestMeasurementControl(unittest.TestCase):

    def test_measurementcontrol(self):
        _ = pyqtgraph.mkQApp()
        with mock.patch('qtt.gui.live_plotting.rda_t'):
            mc = MeasurementControl()
            mc.verbose = 1
            with mock.patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
                mc.enable_measurements()
            std_output = mock_stdout.getvalue()
            self.assertIn('setting qtt_abort_running_measurement to 0', std_output)
            mc.setGeometry(1700, 50, 300, 400)
