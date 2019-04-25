""" Provides the LockInStimulus"""
from qilib.configuration_helper import InstrumentAdapterFactory
from qilib.utils import PythonJsonStructure


class LockInStimulus:
    """ A wrapper that provides methods to control Lock-in of Zurich instrument's UHF-LI devices."""
    def __init__(self, address: str) -> None:
        """ Instantiate a LockInStimulus .
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
        """ Equivalent to enable data transfer in the demodulators section of the Lock-in tab
            in the LabOne web interface.

        Args:
             channel: The channel number to enable (1 - 8).
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

    def set_oscillator_frequency(self, channel: int, frequency: float) -> None:
        """ Set the oscillators frequencies.

        Args:
            channel: Channel the the oscillator belongs to (1 - 8) with MF enabled.
            frequency: Allowed frequencies are 0 - 600 MHz.

        """
        qcodes_parameter_name = 'oscillator{}_freq'.format(channel)
        self._uhfli.parameters[qcodes_parameter_name](frequency)

    def set_signal_output_enabled(self, channel: int, demodulator: int, is_enabled: bool) -> None:
        """ Enable one of the 16 output amplitudes.

        Args:
            channel: The channel to enable (1 - 8).
            demodulator: Which demodulator the channel belongs to.
            is_enabled: True to enable and False to disable.

        """
        qcodes_parameter_name = 'signal_output{}_enable{}'.format(demodulator, channel)
        self._uhfli.parameters[qcodes_parameter_name](is_enabled)

    def set_signal_output_amplitude(self, channel: int, demodulator: int, amplitude: float) -> None:
        """ Set the amplitude of the output signal.

        Args:
            channel: The channel to set the amplitude on (1 - 8).
            demodulator: Which demodulator the channel belongs to.
            amplitude: Amplitude in volts, allowed values are 0.0 - 1.5 V.

        """
        qcodes_parameter_name = 'signal_output{}_amplitude{}'.format(demodulator, channel)
        self._uhfli.parameters[qcodes_parameter_name](amplitude)
