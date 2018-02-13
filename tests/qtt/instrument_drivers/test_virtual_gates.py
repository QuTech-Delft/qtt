import unittest
import numpy as np

from collections import OrderedDict
from qtt.measurements.scans import instrumentName
from qtt.instrument_drivers.virtual_gates import virtual_gates
from qtt.instrument_drivers.virtual_instruments import VirtualIVVI

# -----------------------------------------------------------------------------


class test_virtual_gates(unittest.TestCase):

    def setUp(self):
        gate_names = ['P1', 'P2', 'P3']
        name_ivvi = instrumentName('test_ivvi')
        name_vgates = instrumentName('test_virt_gates')
        self.crosscap_map = OrderedDict((
            ('VP1', OrderedDict((('P1', 1.0), ('P2', 0.6), ('P3', 0.2)))),
            ('VP2', OrderedDict((('P1', 0.3), ('P2', 0.9), ('P3', 0.4)))),
            ('VP3', OrderedDict((('P1', 0.1), ('P2', 0.7), ('P3', 0.8))))))
        self.gates = VirtualIVVI(name_ivvi, model=None, gates=gate_names)
        self.vgates = virtual_gates(name_vgates, self.gates, self.crosscap_map)

    def tearDown(self):
        self.vgates.close()
        self.gates.close()  # part of close vgates???

    def __assertGetSetVirtualPlungerIsEqual(self, vplunger, first: float=5.0,
                                            second: float=10.0):
        self.assertAlmostEqual(vplunger.get(), 0.0)
        vplunger.set(first)
        self.assertAlmostEqual(vplunger.get(), first)
        vplunger.set(second)
        self.assertAlmostEqual(vplunger.get(), second)

    def test_VPs_set_get_HasCorrectValue(self):
        self.__assertGetSetVirtualPlungerIsEqual(self.vgates.VP1)
        self.__assertGetSetVirtualPlungerIsEqual(self.vgates.VP2)
        self.__assertGetSetVirtualPlungerIsEqual(self.vgates.VP3)

    def __assertVirtualPlungersAreEqual(self, expected: dict, actual: dict):
        self.assertEqual(set(expected), set(actual))
        [self.assertAlmostEqual(expected[key], actual[key])
            for key in expected]

    def test_VPs_multi_set_get_HasCorrectValues(self):
        first = OrderedDict({'VP1': 10.1, 'VP2': 20.2, 'VP3': 30.3})
        self.vgates.multi_set(first)
        self.__assertVirtualPlungersAreEqual(self.vgates.allvalues(), first)
        self.vgates.VP1.set(0.0)
        self.vgates.VP2.set(0.0)
        self.vgates.VP3.set(0.0)
        second = {'VP1': -40.4, 'VP2': -50.5, 'VP3': -60.6}
        self.vgates.multi_set(second)
        self.__assertVirtualPlungersAreEqual(self.vgates.allvalues(), second)

    def test_convert_matrix_to_map_HasCorrectValues(self):
        matrix = self.vgates.convert_map_to_matrix(self.crosscap_map)
        self.assertEqual(matrix[0, 0], 1.0)
        self.assertEqual(matrix[0, 1], 0.6)

    def test_convert_identity_HasCorrectValues(self):
        matrix = self.vgates.convert_map_to_matrix(self.crosscap_map)
        crossmap_map_ID = self.vgates.convert_matrix_to_map(matrix)
        self.assertEqual(self.crosscap_map, crossmap_map_ID)

    def test_set_distances(self):
        distances = 1.0 / np.arange(1, 5)
        self.vgates.set_distances(distances)

'''
    vgates=virts.vgates() + ['vP4']
    pgates=virts.pgates() + ['P4']
    virts2= extend_virtual_gates(vgates, pgates, virts, name='vgates')
    if verbose:        
        virts2.print_matrix()    

'''