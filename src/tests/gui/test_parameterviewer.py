from unittest import TestCase
from qtt.gui.parameterviewer import ParameterViewer
import pyqtgraph
import qtt.measurements.scans
from qtt.instrument_drivers.virtual_instruments import VirtualIVVI


class TestGUIParameterViewer(TestCase):

    def test_parameterviewer(self):
        qtapp = pyqtgraph.mkQApp()

        ivvi = VirtualIVVI(name=qtt.measurements.scans.instrumentName('dummyivvi'), model=None)
        p = ParameterViewer(instruments=[ivvi])
        p.show()
        p.updatecallback()
        assert (p.is_running())
        p.setGeometry(10, 10, 360, 600)

        p.set_parameter_properties(minimum_value=0)

        p.stop()
        p.close()
        qtapp.processEvents()
        ivvi.close()
