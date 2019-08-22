import unittest
import pyqtgraph
import time

import qcodes
import qtt.data
import qtt.measurements.scans
import qtt.simulation.virtual_dot_array
from qtt.measurements.videomode_processor import DummyVideoModeProcessor
from qtt.measurements.videomode import VideoMode


class TestVideoModeProcessor(unittest.TestCase):


    def test_DummyVideoModeProcessor(self):
        qtapp = pyqtgraph.mkQApp()

        station = qtt.simulation.virtual_dot_array.initialize()
        dummy_processor = DummyVideoModeProcessor(station)
        vm = VideoMode(station, Naverage=25, diff_dir=None, verbose=2,
                       nplots=1, dorun=False, videomode_processor=dummy_processor)
        vm.stopreadout()
        vm.stop()
        vm.updatebg()
        datasets = vm.get_dataset()
        vm.close()
        qtt.simulation.virtual_dot_array.close()
        self.assertIsInstance(datasets[0], qcodes.DataSet)

        qtapp.processEvents()
        time.sleep(.1)

