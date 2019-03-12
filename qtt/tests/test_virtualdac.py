import unittest
from unittest import TestCase

import qtt.data
import qtt.measurements.scans
from qtt.instrument_drivers.virtual_instruments import VirtualIVVI
from qtt.instrument_drivers.gates import VirtualDAC

# %%


class TestVirtualDAC(TestCase):

    def setUp(self):
        gate_map = {
            'T': (0, 15), 'P1': (0, 3), 'P2': (0, 4),
            'L': (0, 5), 'D1': (0, 6), 'R': (0, 7)}

        self.ivvi = VirtualIVVI(qtt.measurements.scans.instrumentName('ivvi'), model=None)
        self.gates = VirtualDAC(qtt.measurements.scans.instrumentName('gates'),
                                instruments=[self.ivvi], gate_map=gate_map)

    def tearDown(self):
        self.gates.close()
        self.ivvi.close()

    def test_gates_get_set(self):
        expected_value = 100.
        self.gates.R.set(expected_value)
        self.assertEqual(self.gates.R.get(), expected_value)

    def test_named_instruments(self):
        gate_map = {'P1': (0, 1), 'P2': (0, 2), 'P1named': (self.ivvi.name, 1)}

        gates = VirtualDAC(qtt.measurements.scans.instrumentName('gates'), instruments=[self.ivvi], gate_map=gate_map)

        expected_value = 20.
        gates.P1.set(expected_value)
        self.assertEqual(gates.P1named.get(), expected_value)
        gates.close()


if __name__ == '__main__':
    unittest.main()
    pass
