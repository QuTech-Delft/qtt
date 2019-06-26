import unittest
from qcodes import Instrument

from qtt.instrument_drivers.virtualAwg.settings import SettingsInstrument
from qtt.instrument_drivers.virtualAwg.virtual_awg import VirtualAwg


class QuantumDeviceSettings(Instrument):

    def __init__(self):
        super().__init__('settings')
        self.awg_map = {
            'P1': (0, 1),
            'P2': (0, 2),
            'dig_mk': (0, 1, 1)
        }


class TestVirtualAwg(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = SettingsInstrument('')

    def tearDown(self) -> None:
        self.settings.close()

    def test_init_HasNoErrors(self):
        from unittest.mock import Mock
        awg_driver = Mock()
        type(awg_driver).__name__ = 'Tektronix_AWG5014'
        awgs = [awg_driver]

        virtual_awg = VirtualAwg(awgs, self.settings)
        self.assertEqual(virtual_awg.settings, self.settings)
        self.assertEqual(awg_driver, virtual_awg.awgs[0].fetch_awg)
        virtual_awg.close()

    def test_init_HasNoInstruments(self):
        virtual_awg = VirtualAwg(settings=self.settings)
        self.assertEqual(virtual_awg.settings, self.settings)
        self.assertEqual(virtual_awg.instruments, [])
        virtual_awg.close()
