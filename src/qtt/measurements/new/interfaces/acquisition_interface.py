from abc import ABC, abstractmethod
from typing import Any, List

from qilib.data_set import DataSet
from qilib.utils import PythonJsonStructure


class AcquisitionInterface(ABC):

    @abstractmethod
    def __init__(self, address: str) -> None:
        pass

    @abstractmethod
    def initialize(self, configuration: PythonJsonStructure) -> None:
        pass

    @abstractmethod
    def prepare_acquisition(self) -> None:
        pass

    @abstractmethod
    def acquire(self) -> DataSet:
        pass
