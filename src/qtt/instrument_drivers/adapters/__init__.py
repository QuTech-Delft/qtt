from sys import modules

from qilib.configuration_helper.instrument_adapter_factory import InstrumentAdapterFactory
from qtt.instrument_drivers.adapters.time_stamp_instrument_adapter import TimeStampInstrumentAdapter
from qtt.instrument_drivers.adapters.virtual_awg_instrument_adapter import VirtualAwgInstrumentAdapter
from qtt.instrument_drivers.adapters.virtual_dac_instrument_adapter import VirtualDACInstrumentAdapter

InstrumentAdapterFactory.add_instrument_adapter_package(modules['qtt.instrument_drivers.adapters'])
