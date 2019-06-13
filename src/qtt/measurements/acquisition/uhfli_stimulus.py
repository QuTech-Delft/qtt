from typing import Optional, Union

from qcodes import Parameter
from qilib.configuration_helper import InstrumentAdapterFactory
from qilib.utils import PythonJsonStructure


class UHFLIStimulus:
    """ A wrapper that provides methods to control Lock-in of Zurich instrument's UHF-LI devices."""

    def __init__(self, address: str) -> None:
        """ Instantiate a LockInStimulus.

        Args:
            address: A unique ID of a UHFLI device.

        """
        self._adapter = InstrumentAdapterFactory.get_instrument_adapter('ZIUHFLIInstrumentAdapter', address)
        self._uhfli = self._adapter.instrument

    def initialize(self, configuration: PythonJsonStructure) -> None:
        """ Apply configuration to the UHFLI.

        Args:
            configuration: The configuration for the UHFLI.

        """
        self._adapter.apply(configuration)

    def set_demodulation_enabled(self, channel: int, is_enabled: bool) -> None:
        """ Enable data transfer in the demodulators.

        Args:
             channel: The output channel number to enable (1 - 8).
             is_enabled: True to enable and False to disable.

        """
        enabled = 'ON' if is_enabled else 'OFF'
        qcodes_parameter_name = 'demod{}_streaming'.format(channel)
        self._uhfli.parameters[qcodes_parameter_name](enabled)

    def set_output_enabled(self, output: int, is_enabled: bool) -> None:
        """ Control the outputs on the device.

        Args:
            output: One of the two outputs on the device.
            is_enabled: True to enable and False to disable.
        """
        enabled = 'ON' if is_enabled else 'OFF'
        qcodes_parameter_name = 'signal_output{}_on'.format(output)
        self._uhfli.parameters[qcodes_parameter_name](enabled)

    def set_oscillator_frequency(self, oscillator: int, frequency: Optional[float] = None) -> Union[Parameter, None]:
        """ Set the oscillators frequencies.

        Args:
            oscillator: Output channel the the oscillator belongs to (1 - 8) with MF enabled.
            frequency: Optional parameter that if not provided, this method returns a QCoDeS parameter that can be
                    used to set the oscillator frequency. Allowed frequencies are 0 - 600 MHz.
        """
        qcodes_parameter_name = 'oscillator{}_freq'.format(oscillator)
        qcodes_parameter = self._uhfli.parameters[qcodes_parameter_name]
        if frequency is None:
            return qcodes_parameter
        return qcodes_parameter(frequency)

    def set_signal_output_enabled(self, channel: int, demodulator: int, is_enabled: bool) -> None:
        """ Enable one of the 16 output amplitudes.

        Args:
            channel: The output channel to enable (1 - 8).
            demodulator: Which demodulator the channel belongs to.
            is_enabled: True to enable and False to disable.
        """
        qcodes_parameter_name = 'signal_output{}_enable{}'.format(channel, demodulator)
        self._uhfli.parameters[qcodes_parameter_name](is_enabled)

    def set_signal_output_amplitude(self, channel: int, demodulator: int, amplitude: float) -> None:
        """ Set the amplitude of the output signal.

        Args:
            channel: The output channel to set the amplitude on (1 - 8).
            demodulator: Which demodulator the channel belongs to.
            amplitude: Amplitude in volts, allowed values are 0.0 - 1.5 V.
        """
        qcodes_parameter_name = 'signal_output{}_amplitude{}'.format(channel, demodulator)
        self._uhfli.parameters[qcodes_parameter_name](amplitude)

    def set_demodulator_signal_input(self, demodulator: int, input_channel: int) -> None:
        """ Set the signal demodulator input channel.

        Args:
            demodulator: Which demodulator the channel belongs to.
            input_channel: The input channel (1 or 2).
        """
        qcodes_parameter_name = 'demod{}_signalin'.format(demodulator)
        if input_channel not in (1, 2):
            raise NotImplementedError('The input channel can currently only be set to Sig In 1, or Sig In 2!')
        self._uhfli.parameters[qcodes_parameter_name]('Sig In {}'.format(input_channel))

    def connect_oscillator_to_demodulator(self, oscillator: int, demodulator: int) -> None:
        """ Connects an oscillator (1 - 8) to the given demodulator.

        Args:
            oscillator: The oscillator number to connect to (1 - 8).
            demodulator: Which demodulator the oscillator belongs to.
        """
        qcodes_parameter_name = 'demod{}_oscillator'.format(demodulator)
        self._uhfli.parameters[qcodes_parameter_name](oscillator)
