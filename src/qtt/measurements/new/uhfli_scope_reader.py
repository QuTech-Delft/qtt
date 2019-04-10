from typing import Any, List, Optional
import numpy as np
import time

from qilib.configuration_helper import InstrumentAdapterFactory
from qilib.data_set import DataArray, DataSet
from qilib.utils import PythonJsonStructure
from qtt.measurements.new.interfaces import AcquisitionScopeInterface


class UhfliScopeReader(AcquisitionScopeInterface):
    """ Represents an acquisition wrapper for acquiring data with the Zurich Instruments UHFLI."""

    def __init__(self, address):
        self._adapter = InstrumentAdapterFactory.get_instrument_adapter('UhfliInstrumentAdapter', address)
        self.__uhfli = self._adapter.instrument
        self.__acquisition_counter = 0
        self.__address = address

    def initialize(self, config: PythonJsonStructure) -> None:
        self._adapter.apply(config)

    def prepare_acquisition(self) -> None:
        self.__uhfli.scope.set('scopeModule/mode', 1)
        self.__uhfli.scope.subscribe('/{0}/scopes/0/wave'.format(self.__address))
        self.__uhfli.daq.sync()
        self.__uhfli.daq.setInt('/{0}/scopes/0/enable'.format(self.__address), 1)

    def acquire(self, data_set: DataSet, number_of_records: Optional[int]=1, timeout: Optional[int]=30):
        if not data_set.data_arrays:
            self.__add_setpoint_data(data_set)
        trace_data = UhfliScopeReader.get_uhfli_scope_records(self.__address, self.__uhfli,
                                                              number_of_records, timeout)
        self.__acquire_measurement_data(data_set, trace_data)

    def __add_setpoint_data(self, data_set: DataSet):
        self.__acquisition_counter = 0
        sample_count = self.__uhfli.scope_length()
        scope_duration = self.__uhfli.scope_duration()
        data_array = DataArray('ScopeTime', 'Time', unit='seconds', is_setpoint=True,
                                         preset_data=np.linspace(0, scope_duration, sample_count))
        data_set.add_array(data_array)
        data_set.user_data = PythonJsonStructure(sample_rate=self.sample_rate, period=self.period)
 
    def __acquire_measurement_data(self, data_set, trace_data):
        traces = zip(self.__uhfli.Scope.names, self.__uhfli.Scope.units, trace_data)
        for (label, unit, trace) in traces:
            if not isinstance(trace, np.ndarray):
                continue
            self.__acquisition_counter += 1
            identifier = 'ScopeTrace_{:03d}'.format(self.__acquisition_counter)
            data_array = DataArray(identifier, label, unit, preset_data=trace)
            data_set.add_array(data_array)

    @property
    def number_of_averages(self):
        return self.__uhfli.scope_average_weight.get()

    @number_of_averages.setter
    def number_of_averages(self, value):
        self.__uhfli.scope_segments.set('OFF' if value==1 else 'ON')
        self.__uhfli.scope_average_weight.set(value)

    @property
    def input_range(self):
        range_channel_1 = self.__uhfli.signal_input1_range.get()
        range_channel_2 = self.__uhfli.signal_input2_range.get()
        return [range_channel_1, range_channel_2]

    @input_range.setter
    def input_range(self, value):
        range_channel_1, range_channel_2 = value
        self.__uhfli.signal_input1_range.set(range_channel_1)
        self.__uhfli.signal_input2_range.set(range_channel_2)

    @property
    def sample_rate(self):
        return self.__uhfli.scope_samplingrate_float.get()

    @sample_rate.setter
    def sample_rate(self, value):
        self.__uhfli.scope_samplingrate_float.set(value)

    @property
    def period(self):
        return self.__uhfli.scope_duration.get()

    @period.setter
    def period(self, value: float) -> None:
        self.__uhfli.scope_duration.set(value)

    @property
    def trigger_enabled(self) -> bool:
        is_enabled = self.__uhfli.scope_trig_enable.get()
        if is_enabled == 'ON':
            return True
        if is_enabled == 'OFF':
            return False
        raise ValueError('Unknown trigger value ({})!'.format(is_enabled))

    @trigger_enabled.setter
    def trigger_enabled(self, value: bool) -> None:
        self.__uhfli.scope_trig_enable.set('ON' if value else 'OFF')
        self.__check_scope_settings()

    def set_trigger_settings(self, channel: str, level: float, slope: str, delay: float) -> None:
        self.__uhfli.scope_trig_signal.set(channel)
        self.__uhfli.scope_trig_level.set(level)
        self.__uhfli.scope_trig_slope.set(slope)
        self.__uhfli.scope_trig_delay.set(delay)
        self.__uhfli.scope_trig_reference.set(0)

    @property
    def enabled_channels(self):
        enabled_channels = self.__uhfli.scope_channels.get()
        if enabled_channels == 3:
            return [1, 2]
        return [enabled_channels]

    @enabled_channels.setter
    def enabled_channels(self, value):
        if not isinstance(value, list) or len(value) < 1 or len(value) > 2:
            raise ValueError('Invalid enabled channels specification {}!'.format(value))
        self.__uhfli.scope_channels.set(value[0] if value != [1, 2] else 3)

    def set_input_signal(self, channel: int, attribute: str) -> None:
        channel_input = getattr(self.__uhfli, 'scope_channel{}_input'.format(channel))
        channel_input.set(attribute)

    def __check_scope_settings(self):
        self.__uhfli.Scope.prepare_scope()
        if not self.__uhfli.scope_correctly_built:
            raise ValueError('Invalid scope setting! Scope cannot acquire.')

    @staticmethod
    def get_uhfli_scope_records(address, uhfli, number_of_records, timeout):
        uhfli.scope.execute()

        records = 0
        progress = 0
        start = time.time()
        while records < number_of_records or progress < 1.0:
            records = uhfli.scope.getInt("scopeModule/records")
            progress = uhfli.scope.progress()[0]
            if time.time() - start > timeout:
                error_text = "Got {} records after {} sec. - forcing stop.".format(number_of_records, timeout)
                raise TimeoutError(error_text)

        traces = uhfli.scope.read(True)
        wave_nodepath = '/{0}/scopes/0/wave'.format(address)
        return traces[wave_nodepath][:number_of_records][0][0]['wave']

    def finalize_acquisition(self):
        self.__uhfli.daq.setInt('/{0}/scopes/0/enable'.format(self.__address), 0)
        self.__uhfli.scope.finish()
