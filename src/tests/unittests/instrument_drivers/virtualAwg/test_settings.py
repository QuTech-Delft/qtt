from unittest import TestCase

from qtt.instrument_drivers.virtualAwg.settings import SettingsInstrument


class TestSettingsInstrument(TestCase):
    def test_simple(self):
        settings = SettingsInstrument('')
        settings.awg_gates = {'P1': (0, 4)}
        settings.awg_markers = {'m4i_mk': (0, 4, 0)}
        settings.create_map()

        self.assertEqual(settings.awg_map, {'P1': (0, 4), 'm4i_mk': (0, 4, 0)})

        self.assertIn('awg_to_P1', settings.parameters)
        self.assertIn('awg_to_m4i_mk', settings.parameters)

        settings.close()
