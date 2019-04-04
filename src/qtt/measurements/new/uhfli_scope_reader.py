from typing import Any, List, Optional

from qcodes import Measure
from qcodes.instrument_drivers.ZI.ZIUHFLI import ZIUHFLI

from qilib.configuration_helper import InstrumentAdapterFactory
from qilib.data_set import DataSet
from qilib.utils import PythonJsonStructure
from qtt.measurements.new.interfaces import AcquisitionScopeInterface


class UhfliScopeReader(AcquisitionScopeInterface):

    def __init__(self, address):
        self._adapter = InstrumentAdapterFactory.get_instrument_adapter('UhfliInstrumentAdapter', address)
        self.number_of_averages = 10
        self.sampling_rate = 110e3
        self.input_range = 5.0
        self.period = 1e-3

    def initialize(self, config: PythonJsonStructure) -> None:
        self._adapter.apply(config)

    def set_scope_signals(self, channels: List[int], attributes: List[str]) -> None:
        channel_count = len(channels)
        if channel_count != len(attributes) or not (0 < channel_count < 2):
            raise ValueError('Invalid signal inputs! ({}, {})'.format(channels, attributes))

        for channel, attribute in zip(channels, attributes):
            channel_input = getattr(self._adapter.instrument, 'scope_channel{}_input'.format(channel))
            channel_input.set(attribute)

        scope_channels = channels[0] if channel_count == 1 else 3
        self._adapter.instrument.scope_channels.set(scope_channels)

    def set_trigger_enabled(self, is_enabled: bool) -> None:
        is_enabled_as_string = 'ON' if is_enabled else 'OFF'
        self._adapter.instrument.scope_trig_enable.set(is_enabled_as_string)

    def prepare_acquisition(self) -> None:
        uhfli = self._adapter.instrument

        uhfli.scope_mode.set('Time Domain')
        uhfli.scope_samplingrate_float(self.sampling_rate)
        uhfli.scope_duration(self.period)

        uhfli.scope_segments('ON')
        uhfli.scope_average_weight(self.number_of_averages)

        uhfli.Scope.prepare_scope()
        _ = self._adapter.instrument.scope_trig_holdoffseconds.get()

        if not uhfli.scope_correctly_built:
            raise ValueError('Invalid scope setting! Scope cannot acquire.')

    def acquire(self) -> DataSet:
        return Measure(self._adapter.instrument.Scope).run()
