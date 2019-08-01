import unittest
from unittest.mock import MagicMock, patch
import qilib
from qcodes import Instrument
from qilib.configuration_helper import InstrumentAdapterFactory, SerialPortResolver
from qilib.configuration_helper.adapters import CommonInstrumentAdapter
from qilib.utils import PythonJsonStructure
from qtt.instrument_drivers.adapters import VirtualDACInstrumentAdapter


class DummyInstrument(Instrument):
    def __init__(self, name, **kwargs):
        """ Dummy instrument used for testing."""
        super().__init__(name, **kwargs)

        self.add_parameter('amplitude',
                           label='%s amplitude' % name,
                           unit='V',
                           get_cmd=None,
                           set_cmd=None
                           )
        self.add_parameter('frequency',
                           get_cmd=None,
                           set_cmd=None,
                           unit='Hz',
                           label=name)
        self.add_parameter('enable_output',
                           get_cmd=None,
                           set_cmd=None,
                           label=name)

    def get_idn(self):
        idn = {'vendor': 'QuTech', 'model': self.name,
               'serial': 42, 'firmware': '20-05-2019-RC'}
        return idn


class DummyInstrumentAdapter(CommonInstrumentAdapter):
    """ Dummy instrument adapter used for testing. """

    def __init__(self, address: str) -> None:
        super().__init__(address)
        self._instrument = DummyInstrument(name=self.name)

    def _filter_parameters(self, parameters):
        return parameters


class TestVirtualDACInstrumentAdapter(unittest.TestCase):
    def setUp(self):
        SerialPortResolver.serial_port_identifiers['spirack3'] = 'COM3'

    def test_apply(self):
        qilib.configuration_helper.adapters.DummyInstrumentAdapter = DummyInstrumentAdapter
        dummy_adapter = InstrumentAdapterFactory.get_instrument_adapter('DummyInstrumentAdapter', 'some_address')
        dummy_adapter.instrument.amplitude(1)
        dummy_adapter.instrument.frequency(1)
        dummy_adapter.instrument.enable_output(False)
        mock_virtual_dac_instance = MagicMock()
        with patch('qtt.instrument_drivers.adapters.virtual_dac_instrument_adapter.VirtualDAC') as mock_virtual_dac:
            mock_virtual_dac.return_value = mock_virtual_dac_instance
            adapter = VirtualDACInstrumentAdapter('spirack3_module3')
            mock_virtual_dac.assert_called()
        config = PythonJsonStructure()
        config['config'] = snapshot['parameters']
        config['boundaries'] = {'P1': (-10, 10)}
        config['gate_map'] = {'P1': (0, 1)}
        config['instruments'] = {dummy_adapter.name: {'address': 'some_address',
                                                      'adapter_class_name': 'DummyInstrumentAdapter',
                                                      'config': {'amplitude': {'value': 1e-3},
                                                                 'frequency': {'value': 130e6},
                                                                 'enable_output': {'value': True}}}}
        adapter.apply(config)

        mock_virtual_dac_instance.set_boundaries.assert_called_with({'P1': (-10, 10)})
        mock_virtual_dac_instance.add_instruments.assert_called_with([dummy_adapter.instrument])
        self.assertDictEqual({'P1': (0, 1)}, mock_virtual_dac_instance.gate_map)
        self.assertEqual(1e-3, dummy_adapter.instrument.amplitude())
        self.assertEqual(130e6, dummy_adapter.instrument.frequency())
        self.assertTrue(dummy_adapter.instrument.enable_output())

        dummy_adapter.close_instrument()

    def test_read(self):
        qilib.configuration_helper.adapters.DummyInstrumentAdapter = DummyInstrumentAdapter
        dummy_adapter = InstrumentAdapterFactory.get_instrument_adapter('DummyInstrumentAdapter', 'other_address')
        dummy_adapter.instrument.amplitude(42)
        dummy_adapter.instrument.frequency(5)
        dummy_adapter.instrument.enable_output(True)
        mock_virtual_dac_instance = MagicMock()
        mock_virtual_dac_instance.get_boundaries.return_value = {'C1': (-4000, 4000)}
        mock_virtual_dac_instance.snapshot.return_value = snapshot

        bootstrap_config = PythonJsonStructure(boundaries={}, gate_map={'C1': (0, 2)}, config={})
        bootstrap_config['instruments'] = {
            'DummyInstrumentAdapter_other_address': {
                'address': 'other_address',
                'adapter_class_name': 'DummyInstrumentAdapter',
                'config': {}}}

        with patch('qtt.instrument_drivers.adapters.virtual_dac_instrument_adapter.VirtualDAC') as mock_virtual_dac:
            mock_virtual_dac.return_value = mock_virtual_dac_instance
            adapter = VirtualDACInstrumentAdapter('spirack3_module3')
            adapter.apply(bootstrap_config)
        config = adapter.read()
        self.assertIn('C1', config['config'])
        self.assertIn('C2', config['config'])
        self.assertIn('C3', config['config'])
        self.assertIn('rc_times', config['config'])
        self.assertDictEqual(config['boundaries'], {'C1': (-4000, 4000)})
        self.assertDictEqual(config['gate_map'], {'C1': (0, 2)})
        self.assertEqual(config['instruments'][dummy_adapter.name]['address'], 'other_address')
        self.assertEqual(config['instruments'][dummy_adapter.name]['adapter_class_name'], 'DummyInstrumentAdapter')
        self.assertEqual(dummy_adapter.instrument.amplitude(),
                         config['instruments'][dummy_adapter.name]['config']['amplitude']['value'])
        self.assertEqual(dummy_adapter.instrument.frequency(),
                         config['instruments'][dummy_adapter.name]['config']['frequency']['value'])
        self.assertEqual(dummy_adapter.instrument.enable_output(),
                         config['instruments'][dummy_adapter.name]['config']['enable_output']['value'])

        dummy_adapter.close_instrument()


snapshot = {'__class__': 'qtt.instrument_drivers.gates.VirtualDAC',
            'functions': {'get_C1': {},
                          'get_C2': {},
                          'get_C3': {},
                          'set_C1': {},
                          'set_C2': {},
                          'set_C3': {}},
            'name': 'gates0',
            'parameters': {'C1': {'__class__': 'qcodes.instrument.parameter.Parameter',
                                  'full_name': 'gates0_C1',
                                  'instrument': 'qtt.instrument_drivers.gates.VirtualDAC',
                                  'instrument_name': 'gates0',
                                  'inter_delay': 0,
                                  'label': 'C1',
                                  'name': 'C1',
                                  'post_delay': 0,
                                  'raw_value': None,
                                  'ts': None,
                                  'unit': 'mV',
                                  'value': 0},
                           'C2': {'__class__': 'qcodes.instrument.parameter.Parameter',
                                  'full_name': 'gates0_C2',
                                  'instrument': 'qtt.instrument_drivers.gates.VirtualDAC',
                                  'instrument_name': 'gates0',
                                  'inter_delay': 0,
                                  'label': 'C2',
                                  'name': 'C2',
                                  'post_delay': 0,
                                  'raw_value': None,
                                  'ts': None,
                                  'unit': 'mV',
                                  'value': 0},
                           'C3': {'__class__': 'qcodes.instrument.parameter.Parameter',
                                  'full_name': 'gates0_C3',
                                  'instrument': 'qtt.instrument_drivers.gates.VirtualDAC',
                                  'instrument_name': 'gates0',
                                  'inter_delay': 0,
                                  'label': 'C3',
                                  'name': 'C3',
                                  'post_delay': 0,
                                  'raw_value': None,
                                  'ts': None,
                                  'unit': 'mV',
                                  'value': 0},
                           'IDN': {'__class__': 'qcodes.instrument.parameter.Parameter',
                                   'full_name': 'gates0_IDN',
                                   'instrument': 'qtt.instrument_drivers.gates.VirtualDAC',
                                   'instrument_name': 'gates0',
                                   'inter_delay': 0,
                                   'label': 'IDN',
                                   'name': 'IDN',
                                   'post_delay': 0,
                                   'raw_value': None,
                                   'ts': None,
                                   'unit': '',
                                   'vals': '<Anything>',
                                   'value': 0},
                           'rc_times': {'__class__': 'qcodes.instrument.parameter.Parameter',
                                        'full_name': 'gates0_rc_times',
                                        'instrument': 'qtt.instrument_drivers.gates.VirtualDAC',
                                        'instrument_name': 'gates0',
                                        'inter_delay': 0,
                                        'label': 'rc_times',
                                        'name': 'rc_times',
                                        'post_delay': 0,
                                        'raw_value': None,
                                        'ts': None,
                                        'unit': '',
                                        'value': 0}},
            'submodules': {}}
