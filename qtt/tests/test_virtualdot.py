from unittest import TestCase

import qcodes
import qcodes.tests.data_mocks

import qtt.data
import qtt.measurements.scans
from qtt.instrument_drivers.virtual_instruments import VirtualIVVI
from qtt.instrument_drivers.gates import virtual_IVVI as virtual_gates
from qtt.simulation.dotsystem import TripleDot


#%%


class TestVirtualGates(TestCase):

    def setUp(self):
        gate_map = {
            'T': (0, 15), 'P1': (0, 3), 'P2': (0, 4),
            'L': (0, 5), 'D1': (0, 6), 'R': (0, 7)}

        self.ivvi = VirtualIVVI(qtt.measurements.scans.instrumentName('ivvi'), model=None)
        self.gates = virtual_gates(qtt.measurements.scans.instrumentName('gates'), instruments=[self.ivvi], gate_map=gate_map)

    def tearDown(self):
        self.gates.close()
        self.ivvi.close()

    def test_gates(self):
        self.gates.R.set(100)
        self.assertEqual(self.gates.R.get(), 100)


class TestVirtualDot(TestCase):

    def setUp(self):
        self.ds = TripleDot(maxelectrons=2)

    def tearDown(self):
        pass

    def test_dotmodel(self):
        self.ds.calculate_energies({})


if __name__ == '__main__':
    t = TestVirtualGates()
    t = TestVirtualDot()
    pass
