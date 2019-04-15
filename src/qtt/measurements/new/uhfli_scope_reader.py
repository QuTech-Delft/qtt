from typing import List
import numpy as np
import time

from qilib.configuration_helper import InstrumentAdapterFactory
from qilib.data_set import DataArray, DataSet
from qilib.utils import PythonJsonStructure
from qtt.measurements.new.interfaces import AcquisitionScopeInterface
from qcodes.instrument_drivers.ZI import ZIUHFLI


class UhfliScopeReader(AcquisitionScopeInterface):
    """ Represents an acquisition wrapper for acquiring data with the Zurich Instruments UHFLI."""

    def __init__(self, address):
        """ Creates the connection to the UHFLI lock-in amplifier.

        Args:
            address: The unique address of the UHFLI.
        """
        self.adapter = InstrumentAdapterFactory.get_instrument_adapter('ZIUHFLIInstrumentAdapter', address)
        self.__uhfli = self.adapter.instrument
        self.__acquisition_counter = 0
        self.__address = address

    def initialize(self, configuration: PythonJsonStructure) -> None:
        """ Applies the configuration to the UHFLI.

        Args:
            config: The configuration with all the UHFLI settings.
        """
        self.adapter.apply(configuration)

    def prepare_acquisition(self) -> None:
        """ Sets the UFHLI into scope mode such that the device can collect traces."""
        self.__uhfli.scope.set('scopeModule/mode', 1)
        self.__uhfli.scope.subscribe('/{0}/scopes/0/wave'.format(self.__address))
        self.__uhfli.daq.sync()
        self.__uhfli.daq.setInt('/{0}/scopes/0/enable'.format(self.__address), 1)

    def acquire(self, data_set: DataSet, number_of_records: int=1, timeout: float=30) -> None:
        """ Collects traces from the UHFLI, where the readout data is added to the given dataset.

        Args:
            data_set: The object in which all the UHFLI traces are stored on.
            number_of_records: The number of traces which should be collected at once.
            timeout: The time the collecting of traces can maximally take before raising an error.
        """
        if not data_set.data_arrays:
            self.__add_setpoint_data(data_set)
        trace_data = UhfliScopeReader.__get_uhfli_scope_records(self.__address, self.__uhfli,
                                                                number_of_records, timeout)
        self.__acquire_measurement_data(data_set, trace_data)

    def finalize_acquisition(self) -> None:
        """ Disables the acquisition mode of the scope in the UHFLI."""
        self.__uhfli.daq.setInt('/{0}/scopes/0/enable'.format(self.__address), 0)
        self.__uhfli.scope.finish()

    def __add_setpoint_data(self, data_set: DataSet) -> None:
        self.__acquisition_counter = 0
        sample_count = self.__uhfli.scope_length()
        data_array = DataArray('ScopeTime', 'Time', unit='seconds', is_setpoint=True,
                                         preset_data=np.linspace(0, self.period, sample_count))
        data_set.add_array(data_array)
        data_set.user_data = PythonJsonStructure(sample_rate=self.sample_rate, period=self.period)

    def __acquire_measurement_data(self, data_set: DataSet, trace_data: dict) -> None:
        traces = zip(self.__uhfli.Scope.names, self.__uhfli.Scope.units, trace_data)
        for (label, unit, trace) in traces:
            if not isinstance(trace, np.ndarray):
                continue
            self.__acquisition_counter += 1
            identifier = 'ScopeTrace_{:03d}'.format(self.__acquisition_counter)
            data_array = DataArray(identifier, label, unit, preset_data=trace)
            data_set.add_array(data_array)

    @property
    def number_of_averages(self) -> int:
        """ Gets the number of averages to take during a acquisition.

        Returns:
            The number of averages set value.
        """
        return self.__uhfli.scope_average_weight.get()

    @number_of_averages.setter
    def number_of_averages(self, value: int) -> None:
        """ Sets the number of averages to take during a acquisition.

        Args:
            value: The number of averages.
        """
        self.__uhfli.scope_segments.set('OFF' if value==1 else 'ON')
        self.__uhfli.scope_average_weight.set(value)

    @property
    def input_range(self) -> List[float]:
        """ Gets the amplitude input range of the channels.

        Returns:
            The amplitude input range of the channels set value.
        """
        range_channel_1 = self.__uhfli.signal_input1_range.get()
        range_channel_2 = self.__uhfli.signal_input2_range.get()
        return [range_channel_1, range_channel_2]

    @input_range.setter
    def input_range(self, value: List[float]) -> None:
        """ Gets the amplitude input range of the channels.

        Args:
            value: The input range amplitude in Volts.
        """
        range_channel_1, range_channel_2 = value
        self.__uhfli.signal_input1_range.set(range_channel_1)
        self.__uhfli.signal_input2_range.set(range_channel_2)

    @property
    def sample_rate(self) -> float:
        """ Gets the sample rate of the acquisition device.

        Returns:
            The input range amplitude in Volt for each channel.
        """
        return self.__uhfli.scope_samplingrate_float.get()

    @sample_rate.setter
    def sample_rate(self, value: float) -> None:
        """ Sets the sample rate of the acquisition device.

        Args:
            value: The sample rate in samples per second.
        """
        self.__uhfli.scope_samplingrate_float.set(value)

    @property
    def period(self) -> float:
        """ Gets the measuring period of the acquisition.

        Returns:
            The measuring period set value in seconds.
        """
        return self.__uhfli.scope_duration.get()

    @period.setter
    def period(self, value: float) -> None:
        """ Sets the measuring period of the acquisition.

        Args:
            value: The measuring period in seconds.
        """
        self.__uhfli.scope_duration.set(value)

    @property
    def trigger_enabled(self) -> bool:
        """ Gets the external triggering enabled status.

        Returns:
            The trigger enabled status set value.
        """
        is_enabled = self.__uhfli.scope_trig_enable.get()
        if is_enabled == 'ON':
            return True
        if is_enabled == 'OFF':
            return False
        raise ValueError('Unknown trigger value ({})!'.format(is_enabled))

    @trigger_enabled.setter
    def trigger_enabled(self, value: bool) -> None:
        """ Turns the external triggering on or off.

        Args:
            value: The trigger on/off value.
        """
        self.__uhfli.scope_trig_enable.set('ON' if value else 'OFF')

    def set_trigger_settings(self, channel: int, level: float, slope: str, delay: float) -> None:
        """ Updates the input trigger settings.

        Args:
            channel: The channel to trigger the acquision on.
            level: The trigger-level of the trigger.
            slope: The slope of the trigger.
            delay: The delay between getting a trigger and acquiring.
        """
        self.__uhfli.scope_trig_signal.set(channel)
        self.__uhfli.scope_trig_level.set(level)
        self.__uhfli.scope_trig_slope.set(slope)
        self.__uhfli.scope_trig_delay.set(delay)
        self.__uhfli.scope_trig_reference.set(0)

    @property
    def enabled_channels(self) -> List[int]:
        """ Gets the channel enabled states.

        Returns:
            The channels which are set to be enabled.
        """
        enabled_channels = self.__uhfli.scope_channels.get()
        if enabled_channels == 3:
            return [1, 2]
        return [enabled_channels]

    @enabled_channels.setter
    def enabled_channels(self, value: List[int]):
        """ Sets the given channels to enabled and turns off all others.

        Args:
            value: The channels which needs to be anabled.
        """
        if not isinstance(value, list) or len(value) < 1 or len(value) > 2:
            raise ValueError('Invalid enabled channels specification {}!'.format(value))
        self.__uhfli.scope_channels.set(value[0] if value != [1, 2] else 3)

    def set_input_signal(self, channel: int, attribute: str) -> None:
        """ Adds an input channel to the scope.

        Args:
            channel: The input channel number.
            attrbutes: The input signal to acquire.
        """
        channel_input = getattr(self.__uhfli, 'scope_channel{}_input'.format(channel))
        channel_input.set(attribute)

    @staticmethod
    def __get_uhfli_scope_records(address: str, uhfli: ZIUHFLI, number_of_records: int, timeout: float):
        uhfli.scope.execute()

        records = 0
        progress = 0
        start = time.time()
        while records < number_of_records and progress < 1.0:
            records = uhfli.scope.getInt("scopeModule/records")
            progress = uhfli.scope.progress()[0]
            if time.time() - start > timeout:
                error_text = "Got {} records after {} sec. - forcing stop.".format(records, timeout)
                raise TimeoutError(error_text)

        traces = uhfli.scope.read(True)
        wave_nodepath = '/{0}/scopes/0/wave'.format(address)
        return traces[wave_nodepath][:number_of_records][0][0]['wave']
