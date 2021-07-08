from typing import Optional

from qilib.configuration_helper.adapters import CommonInstrumentAdapter
from qilib.utils import PythonJsonStructure

from qtt.instrument_drivers.TimeStamp import TimeStampInstrument
from qtt.utilities.tools import rdeprecated


@rdeprecated(expire='1-1-2022')
class TimeStampInstrumentAdapter(CommonInstrumentAdapter):
    def __init__(self, address: str, instrument_name: Optional[str] = None) -> None:
        super().__init__(address, instrument_name)
        self._instrument = TimeStampInstrument(name=self._instrument_name)

    def _filter_parameters(self, parameters: PythonJsonStructure) -> PythonJsonStructure:
        return parameters
