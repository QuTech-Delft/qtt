""" Provides oscilloscope functionality for the Zurich Instruments UHFLI."""

import time
from typing import List, Tuple, Optional, Union

import numpy as np
from qcodes import Parameter
from qilib.configuration_helper import InstrumentAdapterFactory
from qilib.data_set import DataArray
from qilib.utils import PythonJsonStructure

from qtt.measurements.acquisition.interfaces import AcquisitionScopeInterface


class UHFLIScopeReader(AcquisitionScopeInterface):
    """ Represents an acquisition wrapper for acquiring data with the Zurich Instruments UHFLI."""

    def __init__(self, address):
        """ Creates the connection to the UHFLI lock-in amplifier.

        Args:
            address: The unique address of the UHFLI.
        """
        super().__init__(address)
        self.adapter = InstrumentAdapterFactory.get_instrument_adapter('ZIUHFLIInstrumentAdapter', address)
        self.__uhfli = self.adapter.instrument
        self.__acquisition_counter = 0

    def initialize(self, configuration: PythonJsonStructure) -> None:
        """ Applies the configuration to the UHFLI.

        Args:
            configuration: The configuration with all the UHFLI settings.
        """
        self.adapter.apply(configuration)

    def start_acquisition(self) -> None:
        """ Starts the acquisition mode of the scope in the UHFLI."""
        self.__uhfli.scope.set('scopeModule/mode', 1)
        self.__uhfli.scope.subscribe(f'/{self._address}/scopes/0/wave')
        self.__uhfli.daq.sync()
        self.__uhfli.daq.setInt(f'/{self._address}/scopes/0/enable', 1)

    def acquire(self, number_of_averages: int, timeout: float = 30) -> List[DataArray]:
        """ Collects records from the UHFLI.

        Args:
            number_of_records: The number of records which should be collected at once.
            timeout: The time the collecting of records can maximally take before raising an error.

        Returns:
            A list with the recorded scope records as data arrays.
        """
        self.__create_setpoint_data()
        return self.__get_uhfli_scope_records(number_of_averages, timeout)

    def stop_acquisition(self) -> None:
        """ Stops the acquisition mode of the scope in the UHFLI."""
        self.__uhfli.daq.setInt(f'/{self._address}/scopes/0/enable', 0)
        self.__uhfli.scope.finish()

    def __create_setpoint_data(self) -> None:
        sample_count = self.__uhfli.scope_length()
        data_array = DataArray('ScopeTime', 'Time', unit='s', is_setpoint=True,
                               preset_data=np.linspace(0, self.period, sample_count))
        self.__setpoint_array = data_array

    @property
    def input_range(self) -> Tuple[float, float]:
        """ Gets the amplitude input range of the channels.

        Returns:
            The amplitude input range of the channels set value.
        """
        range_channel_1 = self.__uhfli.signal_input1_range.get()
        range_channel_2 = self.__uhfli.signal_input2_range.get()
        return range_channel_1, range_channel_2

    @input_range.setter
    def input_range(self, value: Tuple[float, float]) -> None:
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
        current_period = self.period
        self.__uhfli.scope_samplingrate_float.set(value)
        self.period = current_period

    def get_nearest_sample_rate(self, sample_rate: float) -> float:
        """ Gets the nearest sample rate corresponding to the given value.

        Args:
            sample_rate: A possible sample rate to check for a nearest actual settable sample rate value.

        Returns:
            The nearest settable sample rate value on the UHFLI.
        """
        return self.__uhfli.round_to_nearest_sampling_frequency(sample_rate)

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
    def number_of_samples(self) -> int:
        """ Gets the sample count to take during a acquisition.

        Returns:
            The number of samples set value.
        """
        return self.__uhfli.scope_length.get()

    @number_of_samples.setter
    def number_of_samples(self, value: int) -> None:
        """ Sets the sample count to take during a acquisition.

        Args:
            value: The number of samples.
        """
        self.__uhfli.scope_length.set(value)

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
        raise ValueError(f'Unknown trigger value ({is_enabled})!')

    @trigger_enabled.setter
    def trigger_enabled(self, value: bool) -> None:
        """ Turns the external triggering on or off.

        Args:
            value: The trigger on/off value.
        """
        self.__uhfli.scope_trig_enable.set('ON' if value else 'OFF')

    @property
    def trigger_channel(self) -> str:
        """ Gets the external triggering channel.

        Returns:
            The trigger channel value. The possible trigger channel values are:
            Signal Input 1, Signal Input 2, Trig Input 1, Trig Input 2, Aux Output 1, Aux Output 2,
            Aux Output 3, Aux Output 4, Aux In 1 Ch 1, Aux In 1 Ch 2, Osc phi Demod 4, Osc phi Demod 8,
            AU Cartesian 1, AU Cartesian 2, AU Polar 1, AU Polar 2.
        """
        return self.__uhfli.scope_trig_signal.get()

    @trigger_channel.setter
    def trigger_channel(self, channel: str) -> None:
        """ Sets the external triggering channel.

        Args:
            channel: The trigger channel value. The possible trigger channel values are:
                     Signal Input 1, Signal Input 2, Trig Input 1, Trig Input 2, Aux Output 1, Aux Output 2,
                     Aux Output 3, Aux Output 4, Aux In 1 Ch 1, Aux In 1 Ch 2, Osc phi Demod 4, Osc phi Demod 8,
                     AU Cartesian 1, AU Cartesian 2, AU Polar 1, AU Polar 2.
        """
        self.__uhfli.scope_trig_signal.set(channel)

    @property
    def trigger_level(self) -> float:
        """ Gets the external triggering level in Volts.

        Returns:
            The trigger level in Volts.
        """
        return self.__uhfli.scope_trig_level.get()

    @trigger_level.setter
    def trigger_level(self, level: float) -> None:
        """ Sets the external triggering level.

        Args:
            level: The external trigger level in Volts.
        """
        self.__uhfli.scope_trig_level.set(level)

    @property
    def trigger_slope(self) -> str:
        """ Gets the external triggering slope.

        Returns:
            The scope trigger slope (possible values are: Rise, Fall, Both or None).
        """
        return self.__uhfli.scope_trig_slope.get()

    @trigger_slope.setter
    def trigger_slope(self, slope: str) -> None:
        """ Sets the external triggering slope.

        Args:
            slope: The external trigger slope (possible values are: Rise, Fall, Both and None).
        """
        self.__uhfli.scope_trig_slope.set(slope)

    @property
    def trigger_delay(self) -> float:
        """ Gets the delay in seconds between the external trigger and acquisition.

        Returns:
            The scope trigger delay in seconds.
        """
        return self.__uhfli.scope_trig_delay.get()

    @trigger_delay.setter
    def trigger_delay(self, delay: float) -> None:
        """ Sets the delay in seconds between the external trigger and acquisition.

        Args:
            delay: The scope trigger delay in seconds.
        """
        self.__uhfli.scope_trig_delay.set(delay)

    @property
    def enabled_channels(self) -> Tuple[int, ...]:
        """ Gets the channel enabled states.

        Returns:
            The channels which are set to be enabled.
        """
        enabled_channels = self.__uhfli.scope_channels.get()
        if enabled_channels == 3:
            return 1, 2
        return enabled_channels,

    @enabled_channels.setter
    def enabled_channels(self, value: Tuple[int, ...]):
        """ Sets the given channels to enabled and turns off all others.

        Args:
            value: The channels which needs to be enabled.
        """
        self.__uhfli.scope_channels.set(sum(value))

    def set_input_signal(self, channel: int, attribute: Optional[str]) -> None:
        """ Adds an input channel to the scope.

        Args:
            channel: The input channel number.
            attribute: The input signal to acquire.
        """
        channel_input = getattr(self.__uhfli, f'scope_channel{channel}_input')
        channel_input.set(attribute)

    def set_channel_limits(self, channel: int, lower_limit: float, upper_limit: float) -> None:
        """ Sets the upper and lower limit of the scope full scale range.

        Args:
            channel: The input channel number.
            lower_limit: The lower limit range.
            upper_limit: The upper limit range.
        """
        self.__uhfli.daq.setDouble(f'/{self._address}/scopes/0/channels/{channel - 1}/limitlower', lower_limit)
        self.__uhfli.daq.setDouble(f'/{self._address}/scopes/0/channels/{channel - 1}/limitupper', upper_limit)

    def acquire_single_sample(self, channel: int, parameter: str, partial: bool = False) -> Union[Parameter, float]:
        """ Collect a single point for each added measurement signal.

        Args:
            channel: Input channel that signal is acquired from.
            parameter: Modulation parameter, 'x', 'y', 'phi' or 'R'
            partial: If True return this method as partial, else acquire a sample and return it.

        Returns:
            This method as a partial method or single sample.

        """
        demod_parameter = getattr(self.__uhfli, f'demod{channel}_{parameter}')
        return demod_parameter if partial else demod_parameter()

    def __get_uhfli_scope_records(self, number_of_averages: int, timeout: float) -> List[DataArray]:
        self.__uhfli.scope.execute()

        records = 0
        progress = 0
        start = time.time()
        while records < number_of_averages or progress < 1.0:
            records = self.__uhfli.scope.getInt('scopeModule/records')
            progress = self.__uhfli.scope.progress()[0]
            if time.time() - start > timeout:
                error_text = f'Got {records} records after {timeout} sec. - forcing stop.'
                raise TimeoutError(error_text)

        traces = self.__uhfli.scope.read(True)
        wave_nodepath = f'/{self._address}/scopes/0/wave'
        scope_traces = traces[wave_nodepath][:number_of_averages]
        return self.__convert_scope_data(scope_traces)

    def __convert_scope_data(self, scope_traces: np.ndarray) -> List[DataArray]:
        data = []
        acquisition_counter = 0
        for channel_index, _ in enumerate(self.enabled_channels):
            channel_data = np.array([trace[0]['wave'][channel_index] for trace in scope_traces])
            averaged_data = np.average(channel_data, axis=0)
            data_array = self.__convert_to_data_array(averaged_data, channel_index, acquisition_counter)
            data.append(data_array)
            acquisition_counter += 1
        return data

    def __convert_to_data_array(self, scope_data: np.ndarray, channel_index: int, counter: int) -> DataArray:
        channel_number = channel_index + 1
        identifier = f'ScopeTrace{counter:03d}_Channel{channel_number}'
        input_parameter = getattr(self.__uhfli, f'scope_channel{channel_number}_input')
        label = input_parameter().replace(' ', '_')
        return DataArray(identifier, label, unit='V', preset_data=scope_data, set_arrays=[self.__setpoint_array])
