from abc import ABC, abstractmethod
from typing import Any, List, Optional

from qtt.measurements.new.interfaces import AcquisitionInterface

class AcquisitionScopeInterface(AcquisitionInterface):

    @abstractmethod
    def set_trigger_enabled(self, is_enabled: bool) -> None:
        pass

    """
    @abstractmethod
    def set_trigger_settings(self, channel: int, level: float, slope: string, delay: float) -> None:
        pass
    """

    @abstractmethod
    def set_scope_signals(self, channels: List[int], attributes: Optional[List[str]] = None) -> None:
        pass

