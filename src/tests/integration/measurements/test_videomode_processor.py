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
        #vm.stopreadout()
        #vm.stop()
        #vm.updatebg()
        datasets = vm.get_dataset()
        time.sleep(.2)
        
        qtapp.processEvents()
        print('TestVideoModeProcessor: vm.close')
        vm.close()
        qtapp.processEvents()
        time.sleep(.2)

        print('TestVideoModeProcessor: virtual_dot_array.close')

        qtt.simulation.virtual_dot_array.close()
        
        print('TestVideoModeProcessor:  final qtapp.processEvents')
        qtapp.processEvents()
        time.sleep(.1)

# https://stackoverflow.com/questions/5339062/python-pyside-internal-c-object-already-deleted
# https://stackoverflow.com/questions/17914960/pyqt-runtimeerror-wrapped-c-c-object-has-been-deleted        
        
        
