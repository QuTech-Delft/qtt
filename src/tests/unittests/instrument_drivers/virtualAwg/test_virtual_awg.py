from typing import List, Any
import unittest

from qtt.instrument_drivers.virtualAwg.settings import SettingsInstrument
from qtt.instrument_drivers.virtualAwg.virtual_awg import VirtualAwg
from unittest.mock import Mock, call


class TestVirtualAwg(unittest.TestCase):

    def setUp(self) -> None:
        self.settings = SettingsInstrument('Fake')

    def tearDown(self) -> None:
        self.settings.close()

    @staticmethod
    def __create_awg_drivers() -> List[Any]:
        awg_driver1, awg_driver2 = Mock(), Mock()
        type(awg_driver2).__name__ = 'Tektronix_AWG5014'
        type(awg_driver1).__name__ = 'Tektronix_AWG5014'
        return [awg_driver1, awg_driver2]

    def test_init_HasNoErrors(self) -> None:
        awgs = TestVirtualAwg.__create_awg_drivers()
        awg_driver1, awg_driver2 = awgs
        virtual_awg = VirtualAwg(awgs, settings=self.settings)

        self.assertEqual(virtual_awg.settings, self.settings)
        self.assertEqual(awg_driver1, virtual_awg.awgs[0].fetch_awg)
        self.assertEqual(awg_driver2, virtual_awg.awgs[1].fetch_awg)
        virtual_awg.close()

    def test_snapshot_includes_settings(self) -> None:
        awgs = TestVirtualAwg.__create_awg_drivers()
        awg_driver1, awg_driver2 = awgs
        virtual_awg = VirtualAwg(awgs, settings=self.settings)
        instrument_snapshot = virtual_awg.snapshot()

        self.assertIn('settings_snapshot', instrument_snapshot['parameters'])
        self.assertDictEqual(instrument_snapshot['parameters']['settings_snapshot']['value'],
                             virtual_awg.settings_snapshot())
        virtual_awg.close()

    def test_init_HasNoInstruments(self) -> None:
        virtual_awg = VirtualAwg(settings=self.settings)

        self.assertEqual(virtual_awg.settings, self.settings)
        self.assertEqual(virtual_awg.instruments, [])
        virtual_awg.close()

    def test_add_instruments(self) -> None:
        awgs = TestVirtualAwg.__create_awg_drivers()
        awg_driver1, awg_driver2 = awgs

        virtual_awg = VirtualAwg(settings=self.settings)
        virtual_awg.add_instruments(awgs)
        self.assertEqual(2, len(virtual_awg.instruments))
        self.assertEqual(awgs, virtual_awg.instruments)

        virtual_awg.add_instruments(awgs)
        self.assertEqual(2, len(virtual_awg.instruments))
        self.assertEqual(awgs, virtual_awg.instruments)

        virtual_awg.close()

    def test_run(self):
        awgs = TestVirtualAwg.__create_awg_drivers()
        awg_driver1, awg_driver2 = awgs
        virtual_awg = VirtualAwg(awgs, settings=self.settings)

        awg_driver1.run.assert_not_called()
        awg_driver2.run.assert_not_called()
        virtual_awg.run()
        awg_driver1.run.assert_called_once()
        awg_driver2.run.assert_called_once()

        virtual_awg.close()

    def test_stop(self):
        awgs = TestVirtualAwg.__create_awg_drivers()
        awg_driver1, awg_driver2 = awgs
        virtual_awg = VirtualAwg(awgs, settings=self.settings)

        awg_driver1.stop.assert_not_called()
        awg_driver2.stop.assert_not_called()
        virtual_awg.stop()
        awg_driver1.stop.assert_called_once()
        awg_driver2.stop.assert_called_once()

        virtual_awg.close()

    def test_reset(self):
        awgs = TestVirtualAwg.__create_awg_drivers()
        awg_driver1, awg_driver2 = awgs
        virtual_awg = VirtualAwg(awgs, settings=self.settings)

        awg_driver1.reset.assert_not_called()
        awg_driver2.reset.assert_not_called()
        virtual_awg.reset()
        awg_driver1.reset.assert_called_once()
        awg_driver2.reset.assert_called_once()

        virtual_awg.close()

    def test_enable_outputs(self):
        self.settings.awg_map = {'P1': (0, 1), 'P2': (0, 2), 'P3': (1, 3), 'm4i_mk': (1, 4, 1)}
        virtual_awg = VirtualAwg(settings=self.settings)
        awgs = TestVirtualAwg.__create_awg_drivers()
        awg_driver1, awg_driver2 = awgs
        virtual_awg._awgs = awgs

        virtual_awg.enable_outputs(['P1', 'P2', 'P3'])
        awg_driver1.enable_outputs.assert_has_calls([call([1]), call([2])])
        awg_driver2.enable_outputs.assert_has_calls([call([(3)]), call([4])])

        virtual_awg.close()
