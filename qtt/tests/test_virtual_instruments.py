import unittest
from unittest import TestCase

import qtt.data
import qtt.measurements.scans
from qtt.instrument_drivers.virtual_instruments import VirtualMeter, VirtualIVVI

# %%


class TestVirtualMeter(TestCase):

    def setUp(self):
        self.virtual_meter = VirtualMeter(qtt.measurements.scans.instrumentName('gates'))

    def tearDown(self):
        self.virtual_meter.close()

    def test_amplitude_parmeter(self):
        self.assertEqual(self.virtual_meter.amplitude.unit, 'a.u.')

        self.assertIsInstance(self.virtual_meter.amplitude(), float)
        with self.assertRaises(AttributeError):
            self.virtual_meter.amplitude.set(2.0)


class TestVirtualIVVI(TestCase):

    def setUp(self):
        self.virtual_ivvi = VirtualIVVI(qtt.measurements.scans.instrumentName('ivvi'), model=None, dac_unit='mV')

    def tearDown(self):
        self.virtual_ivvi.close()

    def test_dac_parameters(self):
        self.assertEqual(self.virtual_ivvi.dac1.unit, 'mV')

        self.assertIsInstance(self.virtual_ivvi.dac5(), (float, int))
        self.virtual_ivvi.dac2.set(1.0)
        self.assertIsInstance(self.virtual_ivvi.dac2(), (float, int))
        self.virtual_ivvi.dac4.set(2.0)
        self.assertEqual(self.virtual_ivvi.dac4(), 2.0)

    def test_dac_validator(self):
        with self.assertRaises(ValueError):
            self.virtual_ivvi.dac1.set(4000)


if __name__ == '__main__':
    unittest.main()
    pass
