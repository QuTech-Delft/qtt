import unittest

from qtt.instrument_drivers.virtualAwg.settings import SettingsInstrument
from qtt.instrument_drivers.virtualAwg.virtual_awg import VirtualAwg
from unittest.mock import Mock


class TestVirtualAwg(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = SettingsInstrument('')

    def tearDown(self) -> None:
        self.settings.close()

    def test_init_HasNoErrors(self):
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

    def test_add_instruments(self):
        awg_driver1, awg_driver2 = Mock(), Mock()
        type(awg_driver2).__name__ = 'Tektronix_AWG5014'
        type(awg_driver1).__name__ = 'Tektronix_AWG5014'
        awgs = [awg_driver1, awg_driver2]

        virtual_awg = VirtualAwg(settings=self.settings)
        virtual_awg.add_instruments(awgs)
        self.assertEqual(2, len(virtual_awg.instruments))
        self.assertEqual(awgs, virtual_awg.instruments)

        virtual_awg.add_instruments(awgs)
        self.assertEqual(2, len(virtual_awg.instruments))
        self.assertEqual(awgs, virtual_awg.instruments)

        virtual_awg.close()
