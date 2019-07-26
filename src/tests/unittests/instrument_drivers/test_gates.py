from unittest import TestCase

from qtt.instrument_drivers.gates import VirtualDAC
from qtt.instrument_drivers.virtual_instruments import VirtualIVVI
from qtt.measurements.scans import instrumentName


class TestVirtualDAC(TestCase):

    def setUp(self):
        gate_map = {
            'T': (0, 15), 'P1': (0, 3), 'P2': (0, 4),
            'L': (0, 5), 'D1': (0, 6), 'R': (0, 7)}

        self.ivvi = VirtualIVVI(instrumentName('ivvi'), model=None)
        self.gates = VirtualDAC(instrumentName('gates'),
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

        gates = VirtualDAC(instrumentName('gates'), instruments=[self.ivvi], gate_map=gate_map)

        expected_value = 20.
        gates.P1.set(expected_value)
        self.assertEqual(gates.P1named.get(), expected_value)
        gates.close()

    def test_set_gate_map(self):
        gate_map = {'P1': (0, 1), 'P2': (0, 2), 'P3': (0, 3)}
        virtual_dac = VirtualDAC(instrumentName('gates'), instruments=[self.ivvi], gate_map=gate_map)
        self.assertDictEqual(gate_map, virtual_dac.gate_map)
        self.assertTrue(hasattr(virtual_dac, 'P1'))
        self.assertTrue(hasattr(virtual_dac, 'P2'))
        self.assertTrue(hasattr(virtual_dac, 'P3'))

        new_gate_map = {'C1': (0, 11), 'C2': (0, 12), 'C3': (0, 13)}
        virtual_dac.gate_map = new_gate_map
        self.assertFalse(hasattr(virtual_dac, 'P1'))
        self.assertFalse(hasattr(virtual_dac, 'P2'))
        self.assertFalse(hasattr(virtual_dac, 'P3'))
        self.assertTrue(hasattr(virtual_dac, 'C1'))
        self.assertTrue(hasattr(virtual_dac, 'C2'))
        self.assertTrue(hasattr(virtual_dac, 'C3'))
        self.assertDictEqual(new_gate_map, virtual_dac.gate_map)

        virtual_dac.close()

    def test_get_boundaries(self):
        gate_map = {'P1': (0, 1), 'P2': (0, 2), 'P3': (0, 3)}
        virtual_dac = VirtualDAC(instrumentName('gates'), instruments=[self.ivvi], gate_map=gate_map)
        gate_boundaries = {'P1': (0, 1), 'P2': (-42, 42), 'P3': (-100, 100)}
        virtual_dac.set_boundaries(gate_boundaries)

        boundaries = virtual_dac.get_boundaries()
        self.assertDictEqual(gate_boundaries, boundaries)

        virtual_dac.close()

    def test_set_dacs(self):
        virtual_dac = VirtualDAC(instrumentName('gates'), instruments=[], gate_map={})
        virtual_dac.add_instruments([self.ivvi])
        self.assertIs(self.ivvi, virtual_dac.instruments[0])

        virtual_dac.close()

    def test_add_instruments(self):
        ivvi1 = VirtualIVVI(instrumentName('ivvi'), model=None)
        ivvi2 = VirtualIVVI(instrumentName('ivv2'), model=None)
        instruments = [ivvi1, ivvi2]
        virtual_dac = VirtualDAC(instrumentName('gates'), instruments=[], gate_map={})

        virtual_dac.add_instruments(instruments)
        self.assertEqual(2, len(virtual_dac.instruments))
        self.assertEqual(instruments, virtual_dac.instruments)

        virtual_dac.add_instruments(instruments)
        self.assertEqual(2, len(virtual_dac.instruments))
        self.assertEqual(instruments, virtual_dac.instruments)

        virtual_dac.close()
        ivvi1.close()
        ivvi2.close()
