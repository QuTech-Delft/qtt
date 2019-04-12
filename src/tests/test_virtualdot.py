import unittest

import qtt.data
import qtt.measurements.scans
from qtt.simulation.dotsystem import TripleDot

from qtt.simulation.virtual_dot_array import generate_configuration

# %%


class TestVirtualDot(unittest.TestCase):

    def setUp(self):
        self.ds = TripleDot(maxelectrons=2)

    def tearDown(self):
        pass

    def test_virtualdot(self):
        _, _, gates, _ = generate_configuration(6)

        station = qtt.simulation.virtual_dot_array.initialize(reinit=True, verbose=0)
        _ = station.keithley1.amplitude()
        _ = station.keithley4.amplitude()

        self.assertTrue('P5' in gates)
        self.assertTrue('SD1a' in gates)
        qtt.simulation.virtual_dot_array.close(verbose=0)

    def test_dotmodel(self):
        self.ds.calculate_energies({})


if __name__ == '__main__':
    unittest.main()
