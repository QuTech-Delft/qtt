import unittest
from qcodes import Instrument
from qtt.instrument_drivers.virtualAwg.virtual_awg import VirtualAwg


class TestVirtualAwg(unittest.TestCase):

    def test_init_HasNoErrors(self):
        from unittest.mock import Mock
        awg_driver = Mock()
        type(awg_driver).__name__ = 'Tektronix_AWG5014'
        awgs = [awg_driver]

        class QuantumDeviceSettings(Instrument):

            def __init__(self):
                super().__init__('settings')
                self.awg_map = {
                    'P1': (0, 1),
                    'P2': (0, 2),
                    'dig_mk': (0, 1, 1)
                }

        settings = QuantumDeviceSettings()
        virtual_awg = VirtualAwg(awgs, settings)
        self.assertEqual(awg_driver, virtual_awg.awgs[0].fetch_awg)
        virtual_awg.close()
        settings.close()
