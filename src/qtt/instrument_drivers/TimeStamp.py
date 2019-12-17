import time
from qcodes import Instrument


class TimeStampInstrument(Instrument):
    """
    Instrument that generates a timestamp
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        self.add_parameter('timestamp', unit='s', get_cmd=self._get_timestamp,
                           docstring='Timestamp based on number of seconds since timestamp offset.')
        self.add_parameter('timestamp_offset', unit='s', get_cmd=None, set_cmd=None, initial_value=0,
                           docstring='Timestamp offset. Default value is 0 which corresponds to the epoch.')
        _ = self.timestamp.get()

    def reset(self):
        """ Reset the timestamp offset such that the timestamps start at 0"""
        self.timestamp_offset(time.time())

    def _get_timestamp(self):
        return time.time() - self.timestamp_offset()
