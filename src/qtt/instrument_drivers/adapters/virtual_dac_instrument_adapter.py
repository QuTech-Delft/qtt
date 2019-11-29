from typing import Dict, Optional

from qilib.configuration_helper import InstrumentAdapterFactory, InstrumentAdapter
from qilib.utils import PythonJsonStructure
from qtt.instrument_drivers.adapters.constants import CONFIG, BOUNDARIES, GATE_MAP, INSTRUMENTS, ADDRESS, \
    ADAPTER_CLASS_NAME
from qtt.instrument_drivers.gates import VirtualDAC


class VirtualDACInstrumentAdapter(InstrumentAdapter):
    """ Adapter for the qtt VirtualDAC."""

    def __init__(self, address: str, instrument_name: Optional[str] = None) -> None:
        super().__init__(address, instrument_name)
        self._instrument = VirtualDAC(self._name, instruments=[], gate_map={})
        self._dac_adapters: Dict[str, VirtualDAC] = {}

    def _filter_parameters(self, parameters: PythonJsonStructure) -> PythonJsonStructure:
        return parameters

    def read(self, update: bool = False) -> PythonJsonStructure:
        """ Additionally reads the gate_map, boundaries and configs of the nested dacs."""

        config = PythonJsonStructure()
        config[CONFIG] = super().read(update)
        config[BOUNDARIES] = self._instrument.get_boundaries()
        config[GATE_MAP] = self._instrument.gate_map
        config[INSTRUMENTS] = PythonJsonStructure()
        for adapter_name, adapter in self._dac_adapters.items():
            config[INSTRUMENTS][adapter_name] = PythonJsonStructure()
            config[INSTRUMENTS][adapter_name][CONFIG] = adapter.read(update)
            config[INSTRUMENTS][adapter_name][ADDRESS] = adapter.address
            config[INSTRUMENTS][adapter_name][ADAPTER_CLASS_NAME] = adapter.__class__.__name__
        return config

    def apply(self, config: PythonJsonStructure) -> None:
        """ Apply config to the virtual dac and all nested dacs."""
        self._instrument.instruments = []
        self._dac_adapters = {}

        instruments = config[INSTRUMENTS]
        for instrument in instruments:
            adapter_class_name = instruments[instrument][ADAPTER_CLASS_NAME]
            address = instruments[instrument][ADDRESS]
            adapter_config = instruments[instrument][CONFIG]
            self.add_adapter_to_instrument_adapter(adapter_class_name, address)
            self._dac_adapters[instrument].apply(adapter_config)
        self._instrument.set_boundaries(config[BOUNDARIES])
        self._instrument.gate_map = config[GATE_MAP]
        super().apply(config[CONFIG])

    def add_adapter_to_instrument_adapter(self, instrument_adapter_class_name: str, address: str,
                                          instrument_name: Optional[str] = None) -> None:
        """ Adds a DAC to the virtual DAC and cache a corresponding instrument adapter.

        Args:
            instrument_adapter_class_name: Name of the InstrumentAdapter subclass.
            address: Address of the physical instrument.
            instrument_name: An optional name for the underlying instrument.
        """

        adapter = InstrumentAdapterFactory.get_instrument_adapter(instrument_adapter_class_name, address,
                                                                  instrument_name)
        if adapter.name not in self._dac_adapters:
            self._dac_adapters[adapter.name] = adapter
        if adapter.instrument not in self.instrument.instruments:
            self.instrument.add_instruments([adapter.instrument])
