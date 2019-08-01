from qilib.configuration_helper import InstrumentAdapter
from qilib.utils import PythonJsonStructure

from qtt.instrument_drivers.TimeStamp import TimeStampInstrument


class TimeStampInstrumentAdapter(InstrumentAdapter):
    def __init__(self, address: str) -> None:
        super().__init__(address)
        self._instrument = TimeStampInstrument(name=self._name)

    def apply(self, config: PythonJsonStructure) -> None:
        """ As there is no configuration to apply, this method is a NOP."""

    def _filter_parameters(self, parameters: PythonJsonStructure) -> PythonJsonStructure:
        """ As there is no configuration to read, this method is a NOP."""

    def read(self, update: bool = False) -> PythonJsonStructure:
        """ Override default read mechanism as this adapter has no real configuration.

        Returns:
            Empty python-json structure.
        """
        return PythonJsonStructure()
